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
import traceback # Added for explicit import if needed, though usually available

# LIME and SHAP imports
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries
import shap

# --- PASTE YOUR ModelLoader, ModelInterpreter, and visualize_attribution classes/functions HERE ---
# (As they are unchanged from your provided version, I'm omitting them for brevity in this response,
# but they are essential for the script to run)

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
                    if framework == 'pytorch' and issubclass(item, nn.Module) and item_name != "Module" and item is not nn.Module: # Avoid picking up nn.Module itself
                        model_class = item
                        break
                    elif framework == 'tensorflow' and issubclass(item, tf.keras.Model) and item_name != "Model" and item is not tf.keras.Model: # Avoid picking up tf.keras.Model
                        model_class = item
                        break
            if model_class is None:
                st.error("No valid model class (nn.Module for PyTorch, tf.keras.Model for TensorFlow) found in the provided code.")
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
            st.info("Ensure the number of classes and model architecture in your code match the saved weights. "
                    "For PyTorch, this often relates to the output size of the final Linear layer. "
                    "For TensorFlow, ensure the model is built (e.g., by defining `input_shape` in the first layer or calling `model.build()`) if shapes are not known beforehand.")
            return None

class ModelInterpreter:
    """Class to handle model interpretation methods for both PyTorch and TensorFlow"""

    def __init__(self, model, framework: str, target_size=(48,48), actual_num_input_channels_for_preprocessing=1):
        self.model = model
        self.framework = framework
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.actual_num_input_channels_for_preprocessing = actual_num_input_channels_for_preprocessing

        if framework == 'pytorch':
            self.model = self.model.to(self.device)
            self.model.eval() # Ensure model is in eval mode for PyTorch

    def preprocess_image(self, image: Image.Image, target_size=None, num_channels=None) -> Union[torch.Tensor, tf.Tensor]:
        """Preprocess image for model input"""
        current_target_size = target_size if target_size else self.target_size
        current_num_channels = num_channels if num_channels is not None else self.actual_num_input_channels_for_preprocessing

        if self.framework == 'pytorch':
            transforms_list = []
            if current_num_channels == 1:
                if image.mode != 'L':
                    transforms_list.append(torchvision.transforms.Grayscale(num_output_channels=1))
            elif current_num_channels == 3:
                if image.mode == 'L':
                    transforms_list.append(torchvision.transforms.Lambda(lambda img: img.convert('RGB')))
                elif image.mode == 'RGBA':
                    image = image.convert('RGB') # Convert RGBA to RGB first

            transforms_list.extend([
                torchvision.transforms.Resize(current_target_size),
                torchvision.transforms.ToTensor()
            ])
            preprocess = torchvision.transforms.Compose(transforms_list)
            return preprocess(image).unsqueeze(0).to(self.device)
        else: # TensorFlow
            if current_num_channels == 1:
                img = image.convert('L').resize(current_target_size) if image.mode != 'L' else image.resize(current_target_size)
            else:
                img = image.convert('RGB').resize(current_target_size) if image.mode != 'RGB' else image.resize(current_target_size)

            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            return tf.expand_dims(img_array, 0)

    def get_prediction(self, image: Image.Image) -> Tuple[Union[torch.Tensor, tf.Tensor], int, float]:
        input_tensor = self.preprocess_image(image)
        if self.framework == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class].item() * 100
        else:
            output = self.model(input_tensor, training=False)
            probabilities = tf.nn.softmax(output)
            predicted_class = tf.argmax(output[0]).numpy()
            confidence = float(probabilities[0, predicted_class]) * 100
        return input_tensor, predicted_class, confidence

    def integrated_gradients(self, input_tensor, target_class: int, steps: int = 50):
        if self.framework == 'pytorch':
            ig = IntegratedGradients(self.model)
            baseline = torch.zeros_like(input_tensor).to(self.device)
            attributions, delta = ig.attribute(input_tensor, baselines=baseline,
                                            target=target_class,
                                            n_steps=steps, return_convergence_delta=True)
            return attributions, delta
        else:
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                predictions = self.model(input_tensor, training=False)
                if target_class >= predictions.shape[1]:
                    st.error(f"Target class {target_class} out of range.")
                    return tf.zeros_like(input_tensor), None
                loss = predictions[:, target_class]
            gradients = tape.gradient(loss, input_tensor)
            if gradients is None:
                st.error("Failed to compute TF gradients (Gradient * Input).")
                return tf.zeros_like(input_tensor), None
            attributions = gradients * input_tensor # This is Gradient * Input, not full IG
            return attributions, None

    def occlusion(self, input_tensor, target_class: int, window_size: int = 5, stride: int = 3):
        if self.framework == 'pytorch':
            occlusion_attr = Occlusion(self.model)
            num_channels = input_tensor.shape[1]
            attributions = occlusion_attr.attribute(input_tensor,
                                            target=target_class,
                                            sliding_window_shapes=(num_channels, window_size, window_size),
                                            strides=(1, stride, stride), # stride for channel dim is 1
                                            baselines=0) # Occlude with zeros
            return attributions
        else: # TensorFlow
            original_output = self.model(input_tensor, training=False)
            if target_class >= original_output.shape[1]:
                st.error(f"Target class {target_class} out of range.")
                return tf.zeros_like(input_tensor)
            original_score = original_output[0, target_class]
            height, width, _ = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3] # NHWC
            attributions_np = np.zeros(input_tensor.shape, dtype=np.float32)

            for h in range(0, height - window_size + 1, stride):
                for w in range(0, width - window_size + 1, stride):
                    occluded_np = input_tensor.numpy().copy()
                    occluded_np[0, h:h+window_size, w:w+window_size, :] = 0 # Occlude all channels
                    occluded_tensor = tf.convert_to_tensor(occluded_np)
                    occluded_output = self.model(occluded_tensor, training=False)
                    score_drop = original_score - occluded_output[0, target_class]
                    attributions_np[0, h:h+window_size, w:w+window_size, :] += score_drop.numpy()

            # Normalize attributions if stride < window_size (due to overlapping windows)
            if stride < window_size:
                overlap_counts = np.zeros_like(attributions_np)
                for h_idx in range(0, height - window_size + 1, stride):
                    for w_idx in range(0, width - window_size + 1, stride):
                        overlap_counts[0, h_idx:h_idx+window_size, w_idx:w_idx+window_size, :] +=1
                attributions_np = np.divide(attributions_np, overlap_counts, where=overlap_counts!=0, out=np.zeros_like(attributions_np))

            return tf.convert_to_tensor(attributions_np)

    def grad_cam(self, input_tensor, target_class: int, layer_name: str):
        if self.framework == 'pytorch':
            try:
                target_layer_module = dict([*self.model.named_modules()])[layer_name]
            except KeyError:
                st.error(f"Layer '{layer_name}' not found in PyTorch model.")
                return None
            grad_cam_attr = LayerGradCam(self.model, target_layer_module)
            attributions = grad_cam_attr.attribute(input_tensor, target=target_class)
            # Upsample to input image size
            if attributions.ndim == 4 and attributions.shape[0] == 1: # (1, 1, H_feat, W_feat)
                attributions = F.interpolate(attributions, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
            return attributions
        else: # TensorFlow
            try:
                conv_layer = self.model.get_layer(layer_name)
            except ValueError:
                st.error(f"Layer '{layer_name}' not found in TensorFlow model.")
                return None

            grad_model = tf.keras.Model(inputs=[self.model.inputs], outputs=[conv_layer.output, self.model.output])
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(input_tensor)
                if target_class >= predictions.shape[1]:
                    st.error(f"Target class {target_class} out of range.")
                    return tf.zeros((1, input_tensor.shape[1], input_tensor.shape[2], 1), dtype=tf.float32) # return shape like input
                loss = predictions[:, target_class]

            grads = tape.gradient(loss, conv_output)
            if grads is None: # Should not happen if loss and conv_output are connected
                st.error("Failed to compute gradients for Grad-CAM (TF).")
                return tf.zeros((1, input_tensor.shape[1], input_tensor.shape[2], 1), dtype=tf.float32)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # Global average pooling
            conv_output_processed = conv_output[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(conv_output_processed) # Remove channel dimension if it's 1
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()) # Normalize
            # Resize heatmap to input image size
            heatmap_resized = tf.image.resize(tf.expand_dims(tf.expand_dims(heatmap,0),-1), [input_tensor.shape[1], input_tensor.shape[2]]) # (1, H, W, 1)
            return heatmap_resized

    def lime_explanation(self, image_for_lime_numpy: np.ndarray, num_lime_features: int, num_lime_samples: int):
        explainer = lime.lime_image.LimeImageExplainer()

        def _predict_fn_for_lime_perturbations(numpy_images_batch): # numpy_images_batch is (N, H, W, C) or (N,H,W)
            if self.framework == 'pytorch':
                self.model.eval()
                processed_tensors = []
                for img_np in numpy_images_batch: # img_np is (H,W,C) or (H,W)
                    # LIME passes images suitable for PIL.Image.fromarray
                    pil_img = Image.fromarray(img_np.astype(np.uint8))
                    img_tensor = self.preprocess_image(pil_img,
                                                       target_size=self.target_size,
                                                       num_channels=self.actual_num_input_channels_for_preprocessing) # (1,C,H,W)
                    processed_tensors.append(img_tensor.squeeze(0)) # (C,H,W)
                batch_tensor = torch.stack(processed_tensors).to(self.device) # (N,C,H,W)
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = F.softmax(outputs, dim=1).cpu().numpy() # (N, num_classes)
                return probabilities
            else: # TensorFlow
                processed_tensors_tf = []
                for img_np in numpy_images_batch:
                    pil_img = Image.fromarray(img_np.astype(np.uint8))
                    img_tensor_tf = self.preprocess_image(pil_img,
                                                          target_size=self.target_size,
                                                          num_channels=self.actual_num_input_channels_for_preprocessing) # (1,H,W,C)
                    processed_tensors_tf.append(img_tensor_tf.numpy().squeeze(0)) # (H,W,C)
                batch_tensor_tf = tf.convert_to_tensor(np.array(processed_tensors_tf)) # (N,H,W,C)
                outputs = self.model(batch_tensor_tf, training=False)
                probabilities = tf.nn.softmax(outputs).numpy() # (N, num_classes)
                return probabilities

        explanation = explainer.explain_instance(
            image=image_for_lime_numpy, # This is the single original image (H,W,C) or (H,W)
            classifier_fn=_predict_fn_for_lime_perturbations,
            top_labels=1, # Explain the top predicted label
            hide_color=0, # Use black to hide superpixels
            num_features=num_lime_features, # Max number of superpixels to highlight
            num_samples=num_lime_samples, # Number of perturbed images to generate
            random_seed=42 # For reproducibility
        )
        return explanation

    # Inside ModelInterpreter class

    def shap_explanation(self, input_tensor: Union[torch.Tensor, tf.Tensor], num_background_samples: int, target_class_idx: int): # Added target_class_idx
        """
        Generates SHAP explanations using GradientExplainer.
        input_tensor: The preprocessed input image tensor (batch size 1).
        num_background_samples: Number of background samples for the explainer.
        target_class_idx: The index of the output class to explain.
        """
        if self.framework == 'pytorch':
            self.model.eval()
            background_data = torch.zeros(num_background_samples, *input_tensor.shape[1:]).to(self.device)

            # --- DEBUGGING ---
            with torch.no_grad():
                model_output_input_debug = self.model(input_tensor)
                print(f"DEBUG (SHAP PyTorch): Model output shape for input_tensor: {model_output_input_debug.shape}")
            # --- END DEBUGGING ---

            explainer = shap.GradientExplainer(self.model, background_data)

            # ***** MODIFICATION: Use output_indices *****
            # This should make shap_values a single array of shape (1, C, H, W) directly
            # if the model output is (1, num_classes)
            try:
                shap_values_for_class = explainer.shap_values(input_tensor, output_indices=target_class_idx)
                # shap_values_for_class should now be a single NumPy array of shape (batch_size_of_input, C, H, W)
            except Exception as e_shap_idx:
                st.warning(f"SHAP explainer.shap_values with output_indices={target_class_idx} failed: {e_shap_idx}. Falling back to default behavior.")
                shap_values_for_class = explainer.shap_values(input_tensor) # Fallback


            # --- DEBUGGING SHAP OUTPUT ---
            if isinstance(shap_values_for_class, list): # Should NOT be a list if output_indices works
                print(f"DEBUG (SHAP PyTorch): shap_values_for_class IS A LIST (unexpected with output_indices) of length {len(shap_values_for_class)}")
                for i, sv_arr in enumerate(shap_values_for_class):
                    print(f"  Item {i} shape: {sv_arr.shape}")
            elif isinstance(shap_values_for_class, np.ndarray):
                print(f"DEBUG (SHAP PyTorch): shap_values_for_class is a NumPy ARRAY with shape {shap_values_for_class.shape}")
            else:
                print(f"DEBUG (SHAP PyTorch): shap_values_for_class is of unexpected type: {type(shap_values_for_class)}")
            # --- END DEBUGGING SHAP OUTPUT ---

            return shap_values_for_class # Return the specific class's SHAP values
        else: # TensorFlow
            background_data = tf.zeros((num_background_samples, *input_tensor.shape[1:]), dtype=input_tensor.dtype)
            # --- DEBUGGING TF ---
            model_output_input_debug_tf = self.model(input_tensor)
            print(f"DEBUG (SHAP TF): Model output shape for input_tensor: {model_output_input_debug_tf.shape}")
            # --- END DEBUGGING TF ---

            explainer = shap.GradientExplainer(self.model, background_data)
            try:
                shap_values_for_class = explainer.shap_values(input_tensor, output_indices=target_class_idx)
            except Exception as e_shap_idx_tf:
                st.warning(f"TF SHAP explainer.shap_values with output_indices={target_class_idx} failed: {e_shap_idx_tf}. Falling back.")
                shap_values_for_class = explainer.shap_values(input_tensor)


            # --- DEBUGGING SHAP OUTPUT TF ---
            if isinstance(shap_values_for_class, list):
                print(f"DEBUG (SHAP TF): shap_values_for_class IS A LIST (unexpected) of length {len(shap_values_for_class)}")
                for i, sv_arr in enumerate(shap_values_for_class):
                    print(f"  Item {i} shape: {sv_arr.shape}")
            elif isinstance(shap_values_for_class, np.ndarray):
                print(f"DEBUG (SHAP TF): shap_values_for_class is a NumPy ARRAY with shape {shap_values_for_class.shape}")
            else:
                print(f"DEBUG (SHAP TF): shap_values_for_class is of unexpected type: {type(shap_values_for_class)}")
            # --- END DEBUGGING SHAP OUTPUT TF ---
            return shap_values_for_class


def visualize_attribution(attributions, title: str):
    if attributions is None:
        st.warning(f"No attributions to visualize for {title}.")
        return None

    # Convert to numpy and handle tensor shapes
    if isinstance(attributions, torch.Tensor):
        attr_np = attributions.squeeze(0).cpu().detach().numpy() # (C, H, W) or (H, W)
        if attr_np.ndim == 3: # If (C,H,W), average over channels for visualization
            attr_np = np.mean(attr_np, axis=0) # (H,W)
    else: # TensorFlow tensor or already numpy
        attr_np = attributions.numpy().squeeze(0) if hasattr(attributions, 'numpy') else attributions.squeeze(0)
        if attr_np.ndim == 3 and attr_np.shape[-1] != 1 : # If (H,W,C), average over channels
            attr_np = np.mean(attr_np, axis=-1) # (H,W)
        elif attr_np.ndim == 3 and attr_np.shape[-1] == 1: # If (H,W,1), squeeze last dim
             attr_np = np.squeeze(attr_np, axis=-1) # (H,W)

    attr_np = np.squeeze(attr_np) # Ensure it's 2D
    if attr_np.ndim != 2:
        st.error(f"Attribution for {title} is not 2D after processing. Shape: {attr_np.shape}")
        return None

    # Normalize for display
    min_val, max_val = np.min(attr_np), np.max(attr_np)
    if max_val - min_val < 1e-9: # Avoid division by zero if flat
        normalized_attributions = np.ones_like(attr_np) * 0.5 if max_val > 1e-9 else np.zeros_like(attr_np)
    else:
        normalized_attributions = (attr_np - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(normalized_attributions, cmap="inferno")
    fig.colorbar(im, ax=ax, label="Normalized Attribution Intensity")
    ax.set_title(title)
    ax.axis("off")
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("🖼️ Model Interpretability Tool")
    st.markdown("Upload your model definition, weights, and an image to understand what your model focuses on.")

    col1, col2 = st.columns(2)

    TARGET_IMG_H, TARGET_IMG_W = 48, 48 # Default target size for preprocessing

    with col1:
        st.header("⚙️ Configuration")
        framework = st.selectbox("Select your framework:", ("pytorch", "tensorflow"), key="framework_select")
        # Default channels for preprocessing based on framework conventions
        default_preprocess_channels = 1 if framework == "pytorch" else 3 # Common for PyTorch (e.g. MNIST) vs TF (e.g. ImageNet-style)
        # However, many PyTorch models also use 3 channels. This is a heuristic.

        st.subheader("Model Parameters (for `__init__`)")
        model_def_in_channels = st.number_input(
            f"Number of Input Channels (for model `__init__`):", min_value=1, step=1, value=default_preprocess_channels,
            help=f"This value will be passed as `in_channels` (PyTorch) or `in_channels_tf` (TensorFlow) to your model's constructor if such an argument exists. Default preprocessing will attempt to create images with {default_preprocess_channels} channel(s)."
        )
        # Determine the number of channels to actually use for preprocessing
        # This allows user to override default if their model expects something different from the heuristic
        actual_preprocess_channels = st.selectbox(
            "Number of channels for image preprocessing:",
            options=[1, 3], index=0 if default_preprocess_channels == 1 else 1,
            help="Ensure this matches what your model expects after preprocessing (e.g., 1 for grayscale, 3 for RGB). This will guide how the uploaded image is converted."
        )

        if model_def_in_channels != actual_preprocess_channels:
            st.warning(
                f"Model `__init__` might expect `{model_def_in_channels}` channels (based on your input above), "
                f"but image preprocessing is set to use `{actual_preprocess_channels}` channels. Ensure these are consistent or that your model handles this."
            )

        num_classes_for_model = st.number_input(
            "Number of Output Classes (for model `__init__` and weights):", min_value=1, step=1, value=10,
            help="Must match your model's output layer size and the number of classes it was trained on."
        )

        st.subheader("Step 1: Define Your Model")
        if framework == 'pytorch':
            default_code = f"""class CustomModel(nn.Module): # PyTorch
    def __init__(self, in_channels={model_def_in_channels}, num_classes={num_classes_for_model}):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.relu1 = nn.ReLU() # IMPORTANT: Use non-inplace ReLU for interpretability methods
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU() # Non-inplace
        self.pool2 = nn.MaxPool2d(2, 2)
        # Calculate flattened size dynamically based on TARGET_IMG_H, TARGET_IMG_W
        # Assuming two MaxPool2d(2,2) layers, H and W are each divided by 4.
        self.fc1 = nn.Linear(32 * ({TARGET_IMG_H}//4) * ({TARGET_IMG_W}//4), 128)
        self.relu3 = nn.ReLU() # Non-inplace
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        return self.fc2(x)
"""
        else: # tensorflow
            default_code = f"""class CustomModel(tf.keras.Model): # TensorFlow
    def __init__(self, num_classes={num_classes_for_model}, in_channels_tf={model_def_in_channels}): # in_channels_tf for TF example
        super(CustomModel, self).__init__()
        # For TF, input_shape in the first layer is common
        self.conv1 = layers.Conv2D(16, 3, padding='same', activation='relu',
                                   input_shape=({TARGET_IMG_H}, {TARGET_IMG_W}, in_channels_tf))
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(num_classes) # Output layer, activation often handled by loss function

    def call(self, x, training=False): # Add training flag
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
"""
        model_code = st.text_area("Define your model class:", default_code, height=350, key="model_code_area")
        st.subheader("Step 2: Upload Weights")
        weights_file = st.file_uploader("Upload weights (.pth for PyTorch, .h5 for TensorFlow)", type=["pth", "h5"], key="weights_uploader")
        st.subheader("Step 3: Upload Image")
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

    with col2:
        st.header("🔍 Interpretation Results")
        if 'model_loaded' not in st.session_state: st.session_state.model_loaded = False
        if 'interpreter' not in st.session_state: st.session_state.interpreter = None

        if model_code and weights_file and not st.session_state.model_loaded:
            with st.spinner("Loading model and weights..."):
                model_class = ModelLoader.load_model_from_code(model_code, framework)
                if model_class:
                    try:
                        # Prepare __init__ parameters
                        init_params = {}
                        sig = inspect.signature(model_class.__init__)
                        if 'in_channels' in sig.parameters: init_params['in_channels'] = model_def_in_channels
                        elif 'in_channels_tf' in sig.parameters: init_params['in_channels_tf'] = model_def_in_channels
                        if 'num_classes' in sig.parameters: init_params['num_classes'] = num_classes_for_model
                        # Add other common params if needed, or let users ensure their __init__ matches
                        # Example: if 'dropout_rate' in sig.parameters: init_params['dropout_rate'] = 0.5

                        model_instance_temp = model_class(**init_params)

                        if framework == 'tensorflow' and not model_instance_temp.built and hasattr(model_instance_temp, 'build'):
                            # Attempt to build the model if it has a build method and isn't built
                            # Use actual_preprocess_channels for the dummy input shape
                            dummy_input_shape = (None, TARGET_IMG_H, TARGET_IMG_W, actual_preprocess_channels)
                            try:
                                model_instance_temp.build(dummy_input_shape)
                            except Exception as e_build:
                                st.info(f"TensorFlow model.build() failed: {e_build}. Model may build on first call with actual data.")

                        model_instance_loaded = ModelLoader.load_weights(model_instance_temp, weights_file, framework)
                        if model_instance_loaded:
                            st.session_state.model_instance = model_instance_loaded
                            st.session_state.interpreter = ModelInterpreter(
                                model_instance_loaded, framework,
                                target_size=(TARGET_IMG_H, TARGET_IMG_W),
                                actual_num_input_channels_for_preprocessing=actual_preprocess_channels # Pass this to interpreter
                            )
                            st.session_state.model_loaded = True
                            st.success("Model and weights loaded successfully!")
                        else:
                            st.session_state.model_loaded = False # Error message shown by load_weights
                    except Exception as e_load:
                        st.error(f"Error instantiating model or loading weights: {str(e_load)}")
                        st.code(traceback.format_exc())
                        st.session_state.model_loaded = False

        if st.session_state.model_loaded and image_file:
            interpreter = st.session_state.interpreter
            model = st.session_state.model_instance

            try:
                pil_image = Image.open(image_file)
                st.image(pil_image, caption='Uploaded Image.', use_column_width=True)

                input_tensor, predicted_class, confidence = interpreter.get_prediction(pil_image)
                st.markdown(f"#### Model Prediction: **Class `{predicted_class}`** with **`{confidence:.2f}%`** confidence.")
                if predicted_class >= num_classes_for_model:
                    st.error(f"Prediction Alert: Predicted class `{predicted_class}` is >= `num_classes={num_classes_for_model}`. "
                             f"Please check the 'Number of Output Classes' setting in the configuration. "
                             f"This can lead to errors in interpretation methods.")


                # Prepare image for LIME/SHAP (expects numpy array, (H,W,C) or (H,W))
                # Use interpreter's actual_num_input_channels_for_preprocessing
                temp_pil_image = pil_image.copy()
                if interpreter.actual_num_input_channels_for_preprocessing == 1:
                    if temp_pil_image.mode != 'L': temp_pil_image = temp_pil_image.convert('L')
                else: # 3 channels
                    if temp_pil_image.mode == 'RGBA': temp_pil_image = temp_pil_image.convert('RGB')
                    elif temp_pil_image.mode == 'L': temp_pil_image = temp_pil_image.convert('RGB')
                # Resize to the target size used by the model
                image_for_lime_shap_numpy = np.array(temp_pil_image.resize(interpreter.target_size, Image.LANCZOS))

                # If 1 channel and numpy array is (H,W,1), LIME might prefer (H,W)
                if image_for_lime_shap_numpy.ndim == 3 and image_for_lime_shap_numpy.shape[-1] == 1 and \
                   interpreter.actual_num_input_channels_for_preprocessing == 1:
                    image_for_lime_shap_numpy = image_for_lime_shap_numpy.squeeze(-1)


                st.subheader("Step 4: Choose Interpretation Method")
                method_type = st.radio("Select method type:",
                                       ("Model-Specific (Gradient-based, Layer-specific)",
                                        "Model-Agnostic (Perturbation-based, Black-box)"),
                                       key="method_type_radio")
                method = None
                if method_type.startswith("Model-Specific"):
                    specific_options = ["Integrated Gradients"]
                    # Check for Conv layers for Grad-CAM
                    if framework == 'pytorch' and any(isinstance(m, nn.Conv2d) for m in model.modules()):
                        specific_options.append("Grad-CAM")
                    elif framework == 'tensorflow' and any(isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv1D, tf.keras.layers.Conv3D)) for l in model.layers):
                        specific_options.append("Grad-CAM")
                    method = st.selectbox("Select model-specific method:", specific_options, key="specific_method_select")
                else: # Model-Agnostic
                    agnostic_options = ["Occlusion", "LIME", "SHAP"]
                    method = st.selectbox("Select model-agnostic method:", agnostic_options, key="agnostic_method_select")

                # --- Interpretation Method Logic with Detailed Explanations ---
                if method == "Integrated Gradients":
                    steps = st.slider("Integration steps:", 20, 200, 50, key="ig_steps_slider")
                    if st.button("Generate IG Interpretation", key="ig_button"):
                        with st.spinner("Computing Integrated Gradients..."):
                            attributions, delta = interpreter.integrated_gradients(input_tensor, predicted_class, steps)
                            fig = visualize_attribution(attributions, f"Integrated Gradients (Steps: {steps})")
                            if fig: st.pyplot(fig)
                        st.markdown("---")
                        st.markdown("### Understanding Integrated Gradients (IG)")
                        st.markdown("""
                        Integrated Gradients (IG) is a feature attribution method that assigns an importance score to each input feature (e.g., pixel) for a given prediction. It satisfies two key axioms: *Sensitivity* (if a feature is changed and prediction changes, its attribution should be non-zero) and *Implementation Invariance* (attributions are identical for functionally equivalent networks). It works by accumulating gradients along a straight path from a **baseline input** (e.g., a black image or random noise) to the **actual input image**.
                        """)
                        if framework == 'pytorch' and delta is not None:
                            st.markdown(f"""
                            - **Convergence Delta:** `{delta.item():.4f}`. This measures the approximation error of IG. It's the difference between the sum of attributions and (prediction_on_input - prediction_on_baseline). A value closer to 0 indicates a more accurate attribution. If high (e.g., > 0.05-0.1), consider increasing 'Integration steps'.
                            """)
                        elif framework == 'tensorflow':
                             st.markdown("""
                            - **Note for TensorFlow:** The current TF implementation here uses **Gradient * Input**, a simpler attribution method. It highlights regions with high gradients scaled by input pixel values. While related, it doesn't have the path integration or convergence delta properties of full IG.
                            """)
                        with st.expander("How to Interpret the Heatmap"):
                            st.markdown("""
                            - **Positive vs. Negative Attributions:** The heatmap shows *normalized* attributions. Brighter regions (e.g., yellow/white in 'inferno' cmap) indicate pixels that contributed *more positively* to the model's decision for the predicted class. Darker regions (e.g., purple/black) had less positive, zero, or even negative impact (though normalization here might obscure negative values if they exist and are small).
                            - **Focus on Salient Regions:** Ideally, the brightest pixels should align with the parts of the object that are semantically important for its class (e.g., a cat's face, a car's wheels).
                            - **Magnitude Matters:** The relative brightness indicates the relative importance. A few very bright spots are more telling than a diffuse glow.
                            """)
                        with st.expander("Potential Issues & How to Improve Your Model"):
                            st.markdown("""
                            - **Spurious Correlations:** If IG highlights background elements or irrelevant artifacts that consistently co-occurred with the class in your training data (e.g., a specific watermark always present with 'dog' images).
                                - *Model Improvement:* Diversify your training data with varied backgrounds and contexts. Use data augmentation that breaks these correlations (e.g., random crops, color jitter, adding/removing distractors).
                            - **Overfitting to Noise/Artifacts:** Model might focus on compression artifacts, image borders, or specific textures not core to the object.
                                - *Model Improvement:* Clean your dataset. Augment by simulating these artifacts or pre-processing to remove them. Regularization (L1/L2, dropout) might help.
                            - **Poor Generalization / Diffuse Attributions:** If attributions are scattered randomly or don't make semantic sense, the model might not have learned robust features.
                                - *Model Improvement:* More diverse/larger dataset. Adjust model capacity (simpler if overfitting, more complex if underfitting). Check learning rate, training epochs.
                            - **Baseline Choice (Advanced):** The choice of baseline can influence attributions. A black image is common, but others (blurred, noisy, average image) can reveal different aspects.
                            """)
                        with st.expander("Actionable Steps / What to Try Next"):
                            st.markdown("""
                            - **Analyze Misclassifications:** Run IG on images your model gets wrong. Does it focus on the wrong thing, or is it confused by ambiguous features?
                            - **Compare Across Classes:** If you have multiple classes, see how attributions differ for the same image when targeting different (plausible) classes.
                            - **Iterate on Data/Model:** Use the insights to guide data augmentation strategies, dataset cleaning, or architectural changes.
                            - **Experiment with `Integration steps` (PyTorch):** More steps can give more precise attributions but take longer.
                            """)

                elif method == "Occlusion":
                    win = st.slider("Occlusion window size:", 2, 20, 5, key="occ_win_slider")
                    stride = st.slider("Occlusion stride:", 1, 10, 3, key="occ_stride_slider")
                    if st.button("Generate Occlusion Interpretation", key="occ_button"):
                        with st.spinner("Computing Occlusion... (this can be slow)"):
                            attributions = interpreter.occlusion(input_tensor, predicted_class, win, stride)
                            fig = visualize_attribution(attributions, f"Occlusion (Window: {win}, Stride: {stride})")
                            if fig: st.pyplot(fig)
                        st.markdown("---")
                        st.markdown("### Understanding Occlusion")
                        st.markdown("""
                        Occlusion is a perturbation-based attribution method. It works by systematically masking (occluding) small patches of the input image with a neutral value (e.g., black or grey) and observing the drop in the model's prediction confidence for the target class. Patches whose occlusion causes a large drop in confidence are considered important.
                        """)
                        with st.expander("How to Interpret the Heatmap"):
                            st.markdown("""
                            - **Importance by Impact:** Brighter regions in the heatmap indicate areas where occluding that patch caused the *largest drop* in the model's confidence for the predicted class. These are the most critical regions according to this method.
                            - **Sensitivity Test:** It directly tests the model's sensitivity to the removal of information in different spatial locations.
                            - **Parameters Influence:**
                                - `Window Size`: Determines the size of the occluded patch. Too small might miss larger features; too large might occlude multiple features at once.
                                - `Stride`: Controls how much the window moves at each step. Smaller strides give finer-grained results but are slower.
                            """)
                        with st.expander("Potential Issues & How to Improve Your Model"):
                            st.markdown("""
                            - **Computational Cost:** Can be very slow, especially with small strides and large images, as it requires many forward passes.
                            - **Choice of Occlusion Value:** Occluding with black might be unnatural if black is a valid feature. Other values (grey, average color) can be tried but add complexity.
                            - **Redundant Features:** If the model has learned multiple redundant features for a class, occluding one might not significantly drop the score if others are still visible. The heatmap might then under-represent the true importance of such features individually.
                                - *Model Improvement:* This could hint that the model isn't as efficient as it could be. Techniques like pruning or encouraging feature diversity might be relevant.
                            - **Focus on Edges/High Frequency:** Sometimes, models become sensitive to edge information. Occluding edges might show high importance.
                                - *Model Improvement:* If this is undesirable, data augmentation focusing on textures or broader shapes (e.g., blurring, style transfer in augmentation) could help.
                            """)
                        with st.expander("Actionable Steps / What to Try Next"):
                            st.markdown("""
                            - **Experiment with Parameters:** Adjust `Window Size` and `Stride` to see how the attribution map changes. Start with a larger window/stride for a quick overview, then refine.
                            - **Compare with Other Methods:** Use Occlusion alongside gradient-based methods (like IG or Grad-CAM) to get a more holistic view. They might highlight different aspects.
                            - **Test Robustness:** If the model is overly sensitive to tiny occlusions of irrelevant parts, it might indicate a lack of robustness.
                                - *Model Improvement:* Augmentation techniques like Cutout or Random Erasing during training can improve robustness to occlusions.
                            """)

                elif method == "Grad-CAM":
                    layers_list = []
                    if framework == 'pytorch': layers_list = [n for n,m in model.named_modules() if isinstance(m, nn.Conv2d) and n] # Only named Conv2D layers
                    else: layers_list = [l.name for l in model.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv1D, tf.keras.layers.Conv3D))] # General Conv layers
                    if not layers_list: st.warning("No suitable Conv layers found for Grad-CAM. Ensure your model has named convolutional layers.")
                    else:
                        layer_name = st.selectbox("Select Conv layer for Grad-CAM:", layers_list, key="gcam_layer_select")
                        if st.button("Generate Grad-CAM", key="gcam_button"):
                            with st.spinner("Computing Grad-CAM..."):
                                attributions = interpreter.grad_cam(input_tensor, predicted_class, layer_name)
                                fig = visualize_attribution(attributions, f"Grad-CAM on '{layer_name}' for class {predicted_class}")
                                if fig: st.pyplot(fig)
                            st.markdown("---")
                            st.markdown(f"### Understanding Grad-CAM (Gradient-weighted Class Activation Mapping) on Layer '{layer_name}'")
                            st.markdown(f"""
                            Grad-CAM produces a coarse localization map highlighting the important regions in the image for predicting a concept (e.g., class). It uses the gradients of the target class score with respect to the feature maps of a chosen convolutional layer. These gradients are global-average-pooled to get weights for each feature map, and a weighted combination of feature maps is computed, followed by a ReLU.
                            The heatmap shows which parts of the *selected layer's feature maps* were most influential for the final decision.
                            """)
                            with st.expander("How to Interpret the Heatmap"):
                                st.markdown("""
                                - **Localization of Evidence:** Brighter areas in the heatmap indicate regions in the image (as seen by the chosen layer) that provided the strongest evidence for the predicted class.
                                - **Layer Choice Matters:**
                                    - *Early Layers (closer to input):* Tend to capture low-level features like edges, textures, and colors. Grad-CAM on these layers will show which of these basic features are important.
                                    - *Later Layers (deeper in network):* Capture more complex, semantic features and object parts. Grad-CAM here often provides better localization of the object itself or its discriminative parts.
                                - **Coarseness:** Grad-CAM heatmaps are inherently coarse because they are derived from the (often downsampled) feature maps of a convolutional layer. They highlight broad regions rather than fine pixel-level details.
                                """)
                            with st.expander("Potential Issues & How to Improve Your Model"):
                                st.markdown("""
                                - **Diffuse Heatmaps:** If the heatmap is very spread out and doesn't focus on specific objects, it might indicate the model isn't localizing well or is using very distributed features.
                                    - *Model Improvement:* Ensure sufficient training, diverse data. Architectural choices (e.g., attention mechanisms, different pooling strategies) might influence localization.
                                - **Focus on Irrelevant Context:** If Grad-CAM highlights background or co-occurring objects rather than the target object.
                                    - *Model Improvement:* Similar to IG, this points to spurious correlations. Diversify training data, use targeted data augmentation (e.g., image segmentation to place objects on varied backgrounds).
                                - **Bias in Intermediate Features:** A layer might consistently activate for biased features (e.g., a specific color cast always present with a certain object in training).
                                    - *Model Improvement:* Data balancing and augmentation are key.
                                """)
                            with st.expander("Actionable Steps / What to Try Next"):
                                st.markdown("""
                                - **Explore Different Layers:** Generate Grad-CAM for various convolutional layers (early, middle, late) to understand how the model's focus evolves from simple features to complex concepts.
                                - **Compare Across Classes:** For a multi-class model, see how Grad-CAM heatmaps change when you target different classes for the same image. Does the model look at different parts for different predictions?
                                - **Correlate with Performance:** If Grad-CAM consistently fails to highlight the correct object for a class the model performs poorly on, it's a strong indicator of a problem with how that class is being learned.
                                - **Guided Backpropagation / Grad-CAM++:** For finer-grained visualizations that combine Grad-CAM's localization with pixel-space gradient details, explore methods like Guided Grad-CAM (though not implemented here).
                                """)

                elif method == "LIME":
                    num_lime_features = st.slider("Number of LIME Superpixels (Features):", 1, 30, 10, key="lime_feat_slider")
                    num_lime_samples = st.slider("Number of LIME Samples (Perturbations):", 100, 5000, 1000, step=100, key="lime_samples_slider")
                    if st.button("Generate LIME Explanation", key="lime_button"):
                        with st.spinner("Generating LIME explanation... (can be slow)"):
                            explanation = interpreter.lime_explanation(image_for_lime_shap_numpy, num_lime_features, num_lime_samples)
                            if explanation:
                                # Get image and mask from LIME
                                # Positive_only=False shows both positive (green) and negative (red) contributions
                                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_lime_features, hide_rest=False)
                                fig_lime, ax_lime = plt.subplots(figsize=(7,7))
                                # LIME temp can be float or int, ensure it's float [0,1] for display if it's not already
                                display_temp = temp / 255.0 if temp.max() > 1.0 and temp.dtype == np.uint8 else temp
                                ax_lime.imshow(mark_boundaries(display_temp, mask))
                                ax_lime.set_title(f"LIME Explanation (Top {num_lime_features} superpixels for class {predicted_class})")
                                ax_lime.axis("off")
                                st.pyplot(fig_lime)
                            st.markdown("---")
                            st.markdown("### Understanding LIME (Local Interpretable Model-agnostic Explanations)")
                            st.markdown("""
                            LIME explains the prediction of any classifier in an interpretable and faithful manner by learning an interpretable model (e.g., linear model) locally around the prediction. For images, LIME first segments the image into "superpixels" (groups of similar, contiguous pixels). It then generates a dataset of perturbed samples by turning some superpixels "on" or "off" (e.g., replacing them with grey). It gets predictions for these samples from the black-box model and then fits a sparse linear model to explain the model's behavior in the vicinity of the instance being explained.
                            """)
                            with st.expander("How to Interpret the Heatmap (Superpixel Overlay)"):
                                st.markdown("""
                                - **Superpixel Importance:** The image shows superpixels. Those highlighted in **green** are the ones that, according to LIME's local linear model, contributed *positively* to the prediction of the target class. Superpixels highlighted in **red** (if `positive_only=False`) contributed *negatively*. Uncolored or less intensely colored superpixels had little to no influence in this local approximation.
                                - **Local Explanation:** It's crucial to remember LIME provides a *local* explanation. It explains why the model made this specific prediction for *this particular instance*, not necessarily how the model behaves globally.
                                - **`Number of LIME Features`:** This controls how many of the most important superpixels are shown.
                                - **`Number of LIME Samples`:** More samples can lead to a more stable and accurate local linear model, but increases computation time.
                                """)
                            with st.expander("Potential Issues & How to Improve Your Model (Indirectly)"):
                                st.markdown("""
                                - **Instability:** LIME explanations can sometimes be unstable, meaning small changes in the input or LIME's parameters (like `num_samples` or the random seed for perturbations) can lead to different superpixels being highlighted.
                                    - *Mitigation:* Increase `num_samples`. Run LIME multiple times to check for consistency.
                                - **Superpixel Segmentation:** The quality of the explanation heavily depends on the initial superpixel segmentation. If superpixels don't align well with semantic parts of the image, the explanation might be less meaningful. (This tool uses a default segmentation).
                                - **Faithfulness vs. Interpretability Trade-off:** LIME tries to be faithful to the complex model locally using a simple model. The simplicity of the explanation (e.g., few features) might trade off some faithfulness.
                                - **Identifying Model Flaws:** If LIME consistently highlights superpixels that are nonsensical or irrelevant to the true class (e.g., background noise, watermarks), it's a strong indication your model might be relying on spurious correlations.
                                    - *Model Improvement:* This is a cue to investigate your training data (bias, artifacts) or model architecture (overfitting). Data augmentation and cleaning are key.
                                """)
                            with st.expander("Actionable Steps / What to Try Next"):
                                st.markdown("""
                                - **Vary Parameters:** Experiment with `Number of LIME Features` and `Number of LIME Samples`. See how the explanation changes.
                                - **Focus on Top Features:** Pay most attention to the superpixels LIME deems most important (brightest green/red).
                                - **Sanity Check:** Do the highlighted superpixels make sense given the image content and the predicted class?
                                - **Compare with Other Methods:** Triangulate LIME's findings with model-specific methods (if available) or other model-agnostic methods like SHAP or Occlusion.
                                """)

                # ... (previous code in main function) ...

                elif method == "SHAP":
                    num_shap_bg_samples = st.slider("Number of SHAP Background Samples (GradientExplainer):", 5, 200, 20, key="shap_bg_slider")
                    if st.button("Generate SHAP Explanation", key="shap_button"):
                        with st.spinner("Generating SHAP explanation (GradientExplainer)... (can be slow for many samples)"):
                            # This can return a list of arrays or a single array
                            shap_output = interpreter.shap_explanation(input_tensor, num_shap_bg_samples)

                            shap_values_for_pred_class = None  # Initialize
                            can_plot_shap = False

                            if shap_output is not None:
                                if isinstance(shap_output, list):
                                    # Multi-output model, shap_output is a list of arrays
                                    if not shap_output:  # Empty list
                                        st.warning("SHAP explanation returned an empty list of values.")
                                    elif predicted_class >= len(shap_output):
                                        st.error(
                                            f"Predicted class {predicted_class} is out of bounds for SHAP values list (length {len(shap_output)}). "
                                            f"This might indicate an issue with the model's output layer or the number of classes."
                                        )
                                    else:
                                        shap_values_for_pred_class = shap_output[predicted_class]
                                        can_plot_shap = True
                                elif isinstance(shap_output, np.ndarray):
                                    # Single-output model (or binary with 1 logit), shap_output is a single array
                                    st.info("SHAP returned a single array of SHAP values. Assuming this corresponds to the model's primary output for the prediction.")
                                    shap_values_for_pred_class = shap_output
                                    can_plot_shap = True
                                else:
                                    st.warning(f"SHAP explanation returned an unexpected data type: {type(shap_output)}. Cannot plot.")
                            else:
                                st.warning("SHAP explanation returned None. Cannot plot.")


                            # ... (inside main function, SHAP method part) ...

                            if can_plot_shap and shap_values_for_pred_class is not None:
                                # shap_values_for_pred_class is a NumPy array
                                # PyTorch expected: (1, C, H, W) or (1, H, W)
                                # TensorFlow expected: (1, H, W, C) or (1, H, W)
                                # Observed problematic PyTorch shape: (1, 1, 48, 48, 7) which is (N, C, H, W, K)

                                st.write(f"Intermediate: shap_values_for_pred_class.shape = {shap_values_for_pred_class.shape}") # For debugging

                                # Handle the unexpected 5D case for PyTorch by averaging the last dimension
                                if framework == 'pytorch' and shap_values_for_pred_class.ndim == 5:
                                    st.warning(f"SHAP values for PyTorch have an unexpected 5D shape: {shap_values_for_pred_class.shape}. Will attempt to average over the last dimension.")
                                    # Assuming (N, C, H, W, K) -> (N, C, H, W) by mean over K
                                    shap_values_for_pred_class = np.mean(shap_values_for_pred_class, axis=-1)
                                    st.write(f"After averaging last dim: shap_values_for_pred_class.shape = {shap_values_for_pred_class.shape}")


                                shap_pixel_values = input_tensor.cpu().numpy() if framework == 'pytorch' else input_tensor.numpy()

                                shap_values_plot_N = None
                                pixel_values_plot_N = None

                                if framework == 'pytorch':
                                    # Now, shap_values_for_pred_class should be (1, C, H, W) or (1, H, W)
                                    if shap_values_for_pred_class.ndim == 4: # NCHW format (e.g., (1, C, H, W))
                                        shap_values_plot_N = np.transpose(shap_values_for_pred_class, (0, 2, 3, 1)) # -> (1, H, W, C)
                                        pixel_values_plot_N = np.transpose(shap_pixel_values, (0, 2, 3, 1))       # -> (1, H, W, C)
                                    elif shap_values_for_pred_class.ndim == 3: # NHW format (e.g. (1, H, W) for grayscale)
                                        shap_values_plot_N = shap_values_for_pred_class
                                        pixel_values_plot_N = shap_pixel_values # Assuming pixel_values also NHW
                                        # If pixel_values is NCHW and C=1, it needs to be reshaped/squeezed appropriately before this point or handled here
                                        if pixel_values_plot_N.ndim == 4 and pixel_values_plot_N.shape[1] == 1: # N1HW
                                            pixel_values_plot_N = np.transpose(pixel_values_plot_N, (0,2,3,1)) # Convert to NHW1
                                    else:
                                        st.error(f"Unexpected SHAP values shape for PyTorch after potential reduction: {shap_values_for_pred_class.shape}")
                                        can_plot_shap = False
                                else: # TensorFlow
                                    # shap_values_for_pred_class: (1, H, W, C) or (1, H, W)
                                    shap_values_plot_N = shap_values_for_pred_class
                                    pixel_values_plot_N = shap_pixel_values
                                    if not (shap_values_for_pred_class.ndim == 4 or shap_values_for_pred_class.ndim == 3):
                                        st.error(f"Unexpected SHAP values shape for TensorFlow: {shap_values_for_pred_class.shape}")
                                        can_plot_shap = False

                                if can_plot_shap and shap_values_plot_N is not None and pixel_values_plot_N is not None:
                                    # Squeeze batch dimension (N=1) as shap.image_plot expects (H,W,C) or (H,W)
                                    shap_values_plot = shap_values_plot_N.squeeze(0)
                                    pixel_values_plot = pixel_values_plot_N.squeeze(0)

                                    # Ensure pixel_values_plot is also appropriately shaped for image_plot
                                    # If it was NHW1 from PyTorch transpose, squeeze the last dim
                                    if pixel_values_plot.ndim == 3 and pixel_values_plot.shape[-1] == 1 and \
                                       (shap_values_plot.ndim == 2 or (shap_values_plot.ndim == 3 and shap_values_plot.shape[-1] != 1)):
                                        pixel_values_plot = pixel_values_plot.squeeze(-1)


                                    # Normalize pixel_values_plot for display if not in [0,1] range
                                    if pixel_values_plot.max() > 1.0001:
                                        pixel_values_plot = pixel_values_plot / 255.0
                                    if pixel_values_plot.min() < -0.0001:
                                        min_val, max_val = pixel_values_plot.min(), pixel_values_plot.max()
                                        if max_val - min_val > 1e-9:
                                            pixel_values_plot = (pixel_values_plot - min_val) / (max_val - min_val)
                                        else:
                                            pixel_values_plot = np.zeros_like(pixel_values_plot) if max_val < 1e-9 else np.ones_like(pixel_values_plot) * 0.5

                                   # In main() under SHAP method, inside the plotting block:
# ... (after shap_values_plot and pixel_values_plot are prepared) ...

                                    st.write(f"Final shap_values_plot.shape for image_plot: {shap_values_plot.shape}")
                                    st.write(f"Final pixel_values_plot.shape for image_plot: {pixel_values_plot.shape}")

                                    # Defensive check for shap.image_plot compatibility
                                    compatible_shapes = False
                                    if shap_values_plot.ndim == 2 and pixel_values_plot.ndim == 2: # Grayscale shap, Grayscale pixel
                                        compatible_shapes = True
                                    elif shap_values_plot.ndim == 3 and pixel_values_plot.ndim == 3 and shap_values_plot.shape[-1] == pixel_values_plot.shape[-1]: # Color shap, Color pixel (same channels)
                                        compatible_shapes = True
                                    elif shap_values_plot.ndim == 3 and shap_values_plot.shape[-1] == 1 and pixel_values_plot.ndim == 2: # Grayscale shap (H,W,1), Grayscale pixel (H,W)
                                        compatible_shapes = True
                                    elif shap_values_plot.ndim == 2 and pixel_values_plot.ndim == 3 and pixel_values_plot.shape[-1] == 1: # Grayscale shap (H,W), Grayscale pixel (H,W,1)
                                        compatible_shapes = True
                                    # Add more cases if needed, e.g., shap (H,W,C) and pixel (H,W) (if pixel is grayscale base for color shap)

                                    if not compatible_shapes:
                                        st.error(f"SHAP values shape ({shap_values_plot.shape}) or pixel values shape ({pixel_values_plot.shape}) is incompatible with shap.image_plot expectations after processing.")
                                    else:
                                        # ***** MODIFICATION: Remove plt_fig, manage figure manually *****
                                        plt.figure(figsize=(8,8)) # Create a new figure explicitly for SHAP
                                        shap.image_plot(shap_values=shap_values_plot,
                                                        pixel_values=pixel_values_plot,
                                                        show=False) # IMPORTANT: Let Streamlit handle showing

                                        fig_shap = plt.gcf() # Get the current figure that SHAP plotted on
                                        # Try to adjust title to avoid overlap
                                        current_ax = fig_shap.get_axes()[0] if len(fig_shap.get_axes()) > 0 else None
                                        if current_ax and current_ax.get_title(): current_ax.set_title("") # Clear SHAP default title
                                        y_suptitle = 0.98 if shap_values_plot.ndim == 2 or (shap_values_plot.ndim == 3 and shap_values_plot.shape[-1] == 1) else 0.93
                                        fig_shap.suptitle(f"SHAP Explanation (GradientExplainer) for class {predicted_class}", fontsize=14, y=y_suptitle)
                                        st.pyplot(fig_shap)
                                        plt.close(fig_shap) # Close the figure to free memory
                                # ... rest of the SHAP method ...
                                elif shap_output is not None and not can_plot_shap:
                                     # Error already shown or warning given about unexpected shape
                                     pass
                            elif shap_output is not None and not can_plot_shap : # An issue was found before plotting stage (e.g. out of bounds, empty list)
                                st.warning("Could not proceed with SHAP plot due to issues with SHAP values structure or indexing.")
                            # The case where shap_output is None is already handled by an st.warning.

                            st.markdown("---")
                            st.markdown("### Understanding SHAP (SHapley Additive exPlanations) using GradientExplainer")
                            # ... (rest of the SHAP markdown explanation)
                            st.markdown("""
                            SHAP (SHapley Additive exPlanations) is a framework for explaining the output of machine learning models by assigning an importance value (SHAP value) to each input feature. These values are based on principles from cooperative game theory (Shapley values).

                            `GradientExplainer` is one of the methods to approximate SHAP values specifically for neural networks. It leverages gradients, similar to methods like Integrated Gradients or SmoothGrad. It computes the expected value of gradients by taking into account a distribution of "background" or "reference" samples (here, all-zero images are used as a common baseline). The SHAP values then indicate how much each input pixel pushed the model's output away from the average prediction over this background distribution, towards the actual prediction for the given input.
                            """)
                            with st.expander("How to Interpret the Heatmap"):
                                st.markdown("""
                                - **Red vs. Blue:**
                                    - **Red pixels:** Indicate features (pixels) that *pushed the model's prediction score higher* (more towards the predicted class, or higher probability for that class) relative to the baseline.
                                    - **Blue pixels:** Indicate features that *pushed the model's prediction score lower* (away from the predicted class, or lower probability) relative to the baseline.
                                - **Intensity Matters:** The intensity of the color (brighter red or brighter blue) signifies the magnitude of that pixel's contribution.
                                - **Baseline Comparison:** SHAP values explain the difference between the current prediction and a baseline prediction (implicitly defined by the background data provided to GradientExplainer).
                                - **Additive Property (Theoretical):** SHAP values for all features (pixels) sum up to the difference between the model's output for the instance and the baseline/expected output.
                                """)
                            with st.expander("Potential Issues & How to Improve Your Model (Indirectly)"):
                                st.markdown("""
                                - **Computational Cost:** While `GradientExplainer` is generally faster than `KernelExplainer`, it still processes background samples and computes gradients, which can be intensive for very large models or many background samples.
                                - **Choice of Background/Baseline Data:** This is CRUCIAL for `GradientExplainer`. The SHAP values are relative to this baseline. Using all-zeros (as done here) is a common choice for images. Other options include random noise or a representative subset of training data, which can yield different insights.
                                - **Approximation Quality:** `GradientExplainer` is an approximation of true Shapley values. Its accuracy can depend on factors like the model's linearity in local regions and the representativeness of the background data.
                                - **Sensitivity to Gradients:** Being gradient-based, it can be affected by issues like noisy or vanishing gradients if the model suffers from them, although the expectation over background samples helps.
                                - **Identifying Model Flaws:** If SHAP highlights pixels that are clearly irrelevant (e.g., noise, unimportant background) as strongly contributing (either red or blue), it suggests the model might be learning spurious correlations.
                                    - *Model Improvement:* This signals a need to review training data for biases, improve data augmentation, or potentially adjust the model.
                                """)
                            with st.expander("Actionable Steps / What to Try Next"):
                                st.markdown("""
                                - **Vary Background Samples:** Experiment with the `Number of SHAP Background Samples`. More samples can lead to more stable SHAP values but increase computation time.
                                - **Examine Contributing Features:** Focus on the brightest red and blue pixels. Do they align with semantically meaningful parts of the image for the predicted class?
                                - **Compare Across Predictions:** Analyze SHAP explanations for different images, especially correct vs. incorrect predictions, to understand patterns.
                                - **Compare with Other Explainers:** Use in conjunction with methods like Integrated Gradients (which is conceptually related) or LIME to get a more comprehensive understanding.
                                """)

            except RuntimeError as e_runtime:
                if framework == "pytorch" and ("Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace" in str(e_runtime) or "one of the variables needed for gradient computation has been modified by an inplace operation" in str(e_runtime)):
                    st.error(
                        "**PyTorch Inplace Operation Error Detected!**\n\n"
                        "This error typically occurs because your PyTorch model definition uses 'inplace' operations (e.g., `nn.ReLU(inplace=True)` or `x.relu_()`). "
                        "Interpretability methods like SHAP, Integrated Gradients, and Grad-CAM use hooks that can conflict with inplace modifications of tensors involved in gradient computation.\n\n"
                        "**How to Fix:**\n"
                        "1.  **Review Your Model Code:** Carefully check the PyTorch model class definition you provided.\n"
                        "2.  **Remove `inplace=True`:** Change all instances of `nn.ReLU(inplace=True)` to `nn.ReLU()`. Similarly, replace other inplace functions (like `x.add_()`, `x.mul_()`) with their out-of-place equivalents (e.g., `x = x + ...`, `x = x * ...` or `x = torch.add(x, ...)`).\n"
                        "3.  **Re-upload & Retry:** After modifying your model code, paste the corrected version, re-upload weights, and try the interpretation again."
                        f"\n\n*Detailed error:*\n```\n{str(e_runtime)}\n```"
                    )
                else:
                    st.error(f"A RuntimeError occurred during interpretation: {str(e_runtime)}")
                    st.code(traceback.format_exc())
            except Exception as e_general:
                st.error(f"An error occurred during image processing or interpretation: {str(e_general)}")
                st.code(traceback.format_exc())

        elif st.session_state.model_loaded and not image_file:
            st.info("Model loaded. Please upload an image to proceed with interpretation.")
        elif not model_code or not weights_file:
             st.info("Please define your model and upload weights to begin.")


if __name__ == "__main__":
    main()