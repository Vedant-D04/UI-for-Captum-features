import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import torchvision
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import matplotlib.pyplot as plt
from typing import Union, Tuple
import importlib.util
import inspect

class ModelLoader:
    """Helper class to load and validate ML models dynamically"""
    
    @staticmethod
    def validate_pytorch_model(model_class) -> bool:
        """Validate if the provided class is a valid PyTorch model"""
        return isinstance(model_class, type) and issubclass(model_class, nn.Module)
    
    @staticmethod
    def validate_tensorflow_model(model_class) -> bool:
        """Validate if the provided class is a valid TensorFlow model"""
        return isinstance(model_class, type) and issubclass(model_class, tf.keras.Model)
    
    @staticmethod
    def load_model_from_code(model_code: str, framework: str) -> Union[type, None]:
        """Load model class from provided code string"""
        try:
            temp_module = type(sys)(framework + '_model')
            
            if framework == 'pytorch':
                imports = """
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
            else:
                imports = """
import tensorflow as tf
from tensorflow.keras import layers, Model
"""
            
            exec(imports + model_code, temp_module.__dict__)
            
            model_class = None
            for item_name, item in temp_module.__dict__.items():
                if isinstance(item, type):
                    if framework == 'pytorch' and issubclass(item, nn.Module):
                        model_class = item
                        break
                    elif framework == 'tensorflow' and issubclass(item, tf.keras.Model):
                        model_class = item
                        break
            
            return model_class
            
        except Exception as e:
            st.error(f"Error loading model code: {str(e)}")
            return None

    @staticmethod
    def load_weights(model: Union[nn.Module, tf.keras.Model], 
                    weights_file: tempfile._TemporaryFileWrapper,
                    framework: str) -> Union[nn.Module, tf.keras.Model]:
        """Load weights into the model"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5' if framework == 'tensorflow' else '.pth') as tmp_file:
                tmp_file.write(weights_file.getvalue())
                tmp_path = tmp_file.name
            
            if framework == 'pytorch':
                state_dict = torch.load(tmp_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
            else:
                model.load_weights(tmp_path)
            
            os.unlink(tmp_path)
            return model
            
        except Exception as e:
            st.error(f"Error loading weights: {str(e)}")
            return None

class ModelInterpreter:
    """Class to handle model interpretation methods for both PyTorch and TensorFlow"""
    
    def __init__(self, model, framework: str):
        self.model = model
        self.framework = framework
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if framework == 'pytorch':
            self.model = self.model.to(self.device)
    
    def preprocess_image(self, image: Image.Image, target_size=(48,48)) -> Union[torch.Tensor, tf.Tensor]:
        """Preprocess image for model input"""
        if self.framework == 'pytorch':
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.Resize(target_size),
                torchvision.transforms.ToTensor()
            ])
            return preprocess(image).unsqueeze(0).to(self.device)
        else:
            # TensorFlow preprocessing
            img = image.resize(target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
            return tf.expand_dims(img_array, 0)
    
    def get_prediction(self, image: Image.Image) -> Tuple[Union[torch.Tensor, tf.Tensor], int, float]:
        """Get model prediction for image"""
        input_tensor = self.preprocess_image(image)
        
        if self.framework == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item() 
                confidence = probabilities[0, predicted_class].item() * 100
        else:
            output = self.model(input_tensor)
            probabilities = tf.nn.softmax(output)
            predicted_class = tf.argmax(output[0]).numpy()
            confidence = float(probabilities[0, predicted_class]) * 100
            
        return input_tensor, predicted_class, confidence
    
    def integrated_gradients(self, input_tensor, target_class: int, steps: int = 50):
        """Compute Integrated Gradients attribution"""
        if self.framework == 'pytorch':
            ig = IntegratedGradients(self.model)
            attributions, delta = ig.attribute(input_tensor, target=target_class,
                                            n_steps=steps, return_convergence_delta=True)
            return attributions, delta
        else:
            # TensorFlow Integrated Gradients implementation
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                predictions = self.model(input_tensor)
                loss = predictions[:, target_class]
            
            gradients = tape.gradient(loss, input_tensor)
            attributions = gradients * input_tensor
            return attributions, None
    
    def occlusion(self, input_tensor, target_class: int, window_size: int = 5, stride: int = 3):
        """Compute Occlusion attribution"""
        if self.framework == 'pytorch':
            occlusion = Occlusion(self.model)
            attributions = occlusion.attribute(input_tensor,
                                            target=target_class,
                                            sliding_window_shapes=(1, window_size, window_size),
                                            strides=(3, stride, stride))
            return attributions
        else:
            # TensorFlow Occlusion implementation
            original_output = self.model(input_tensor)
            original_score = original_output[0, target_class]
            
            attributions = tf.zeros_like(input_tensor)
            for h in range(0, input_tensor.shape[1] - window_size + 1, stride):
                for w in range(0, input_tensor.shape[2] - window_size + 1, stride):
                    occluded = input_tensor.numpy().copy()
                    occluded[0, h:h+window_size, w:w+window_size, :] = 0
                    occluded_output = self.model(tf.convert_to_tensor(occluded))
                    attributions[0, h:h+window_size, w:w+window_size, :] = (
                        original_score - occluded_output[0, target_class]
                    )
            
            return attributions
    
    def grad_cam(self, input_tensor, target_class: int, layer_name: str):
        """Compute Grad-CAM attribution"""
        if self.framework == 'pytorch':
            target_layer = dict([*self.model.named_modules()])[layer_name]
            grad_cam = LayerGradCam(self.model, target_layer)
            attributions = grad_cam.attribute(input_tensor, target=target_class)
            return attributions
        else:
            # TensorFlow Grad-CAM implementation
            grad_model = tf.keras.Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(input_tensor)
                loss = predictions[:, target_class]
            
            gradients = tape.gradient(loss, conv_output)
            pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
            
            conv_output = conv_output[0]
            for i in range(pooled_gradients.shape[-1]):
                conv_output[:, :, i] *= pooled_gradients[i]
            
            heatmap = tf.reduce_mean(conv_output, axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return tf.expand_dims(tf.expand_dims(heatmap, 0), -1)

def visualize_attribution(attributions, title: str):
    """Helper function to visualize attributions"""
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.squeeze().cpu().detach().numpy()
    else:
        attributions = attributions.numpy().squeeze()
    
    if attributions.ndim == 3:
        attributions = np.mean(attributions, axis=0)
    
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attributions, cmap="inferno")
    plt.colorbar(label="Attribution Intensity")
    plt.axis("off")
    plt.title(title)
    return plt

def get_model_parameters(model_class):
    """Analyze the model's __init__ method to identify required parameters"""
    signature = inspect.signature(model_class.__init__)
    # Exclude 'self' from the parameters
    params = [param for param_name, param in signature.parameters.items() 
              if param_name != 'self']
    return params

def main():
    st.title("Model Interpretability Tool")
    
    # Framework selection and model input
    framework = st.selectbox("Select your framework:", ("pytorch", "tensorflow"))
    input_channels = st.number_input("Enter the number of input channels:", min_value=1, step=1, value=3)
    
    st.header("Step 1: Define Your Model")
    if framework == 'pytorch':
        default_code = f"""class CustomModel(nn.Module):
    def __init__(self, in_channels={input_channels}):
        super(CustomModel, self).__init__()
        # Define your layers here
        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 111 * 111, 1000)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 64 * 111 * 111)
        x = self.fc1(x)
        return x
"""
    else:
        default_code = f"""class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = layers.Conv2D(64, 3, activation='relu', input_shape=({input_channels}, 224, 224))
        self.pool = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1000)
        
    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.flatten(x)
        return self.fc1(x)
"""
    
    model_code = st.text_area("Define your model class:", default_code, height=300)
    
    if st.button("Submit Model"):
        st.write("Model submitted!")
    
    st.header("Step 2: Upload Weights")
    weights_file = st.file_uploader(
        "Upload weights file",
        type=["pth", "h5"] if framework == "pytorch" else ["h5"]
    )
    
    if model_code and weights_file:
        # Load model
        model_class = ModelLoader.load_model_from_code(model_code, framework)
        if model_class is not None:
            try:
                # Check if the model requires parameters in __init__
                params = get_model_parameters(model_class)
                
                # Initialize model with appropriate parameters
                if framework == 'pytorch':
                    # Check if model requires in_channels parameter
                    if 'in_channels' in inspect.signature(model_class.__init__).parameters:
                        model = model_class(in_channels=input_channels)
                    else:
                        # Try to initialize with no parameters first
                        try:
                            model = model_class()
                        except TypeError as e:
                            # If initialization fails, show detailed error to help user fix their model
                            st.error(f"Error initializing model: {str(e)}")
                            st.info("Your model class seems to require parameters during initialization. " +
                                   "Please check your model definition and make sure all required parameters " +
                                   "have default values or are provided in the code.")
                            return
                else:
                    # TensorFlow model initialization
                    model = model_class()

                model = ModelLoader.load_weights(model, weights_file, framework)
                
                if model is not None:
                    st.success("Model loaded successfully!")
                    
                    # Create interpreter instance
                    interpreter = ModelInterpreter(model, framework)
                    
                    # Image upload
                    st.header("Step 3: Upload Image")
                    image_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
                    
                    if image_file is not None:
                        image = Image.open(image_file)
                        st.image(image, caption='Uploaded Image.', use_container_width=True)
                        
                        # Get prediction
                        input_tensor, predicted_class, confidence = interpreter.get_prediction(image)
                        st.write(f"Predicted class: {predicted_class} with confidence {confidence:.2f}%")
                        
                        # Interpretation method selection
                        st.header("Step 4: Choose Interpretation Method")
                        method = st.selectbox(
                            "Select interpretation method:",
                            ("Integrated Gradients", "Occlusion", "Grad-CAM")
                        )
                        
                        if method == "Integrated Gradients":
                            steps = st.slider("Integration steps:", 20, 200, 50)
                            if st.button("Generate Interpretation"):
                                with st.spinner("Computing attributions..."):
                                    attributions, delta = interpreter.integrated_gradients(
                                        input_tensor, predicted_class, steps
                                    )
                                    plt = visualize_attribution(
                                        attributions,
                                        f"Integrated Gradients (Steps: {steps})"
                                    )
                                    st.pyplot(plt)
                                    if delta is not None:
                                        st.info(f"Convergence delta: {delta.item():.6f}")
                        
                        elif method == "Occlusion":
                            window_size = st.slider("Window size:", 2, 12, 5)
                            stride = st.slider("Stride:", 1, 5, 3)
                            if st.button("Generate Interpretation"):
                                with st.spinner("Computing attributions..."):
                                    attributions = interpreter.occlusion(
                                        input_tensor, predicted_class, window_size, stride
                                    )
                                    plt = visualize_attribution(
                                        attributions,
                                        f"Occlusion (Window: {window_size}, Stride: {stride})"
                                    )
                                    st.pyplot(plt)
                        
                        elif method == "Grad-CAM":
                            # Get available layers
                            if framework == 'pytorch':
                                layers = [name for name, _ in model.named_modules() if name]
                            else:
                                layers = [layer.name for layer in model.layers]
                            
                            layer_name = st.selectbox("Select layer:", layers)
                            if st.button("Generate Interpretation"):
                                with st.spinner("Computing attributions..."):
                                    attributions = interpreter.grad_cam(
                                        input_tensor, predicted_class, layer_name
                                    )
                                    plt = visualize_attribution(
                                        attributions,
                                        f"Grad-CAM: {layer_name}"
                                    )
                                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error instantiating model: {str(e)}")
                st.info("Try modifying your model code to ensure all required parameters have default values.")

if __name__ == "__main__":
    main()