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
                # For TF, if the model needs to be built and input shape isn't defined in __init__,
                # it might fail here if not built before load_weights.
                # We attempt a build before calling load_weights in the main logic.
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

    def __init__(self, model, framework: str):
        self.model = model
        self.framework = framework
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if framework == 'pytorch':
            self.model = self.model.to(self.device)

    def preprocess_image(self, image: Image.Image, target_size=(48,48), num_channels=1) -> Union[torch.Tensor, tf.Tensor]:
        """Preprocess image for model input"""
        if self.framework == 'pytorch':
            transforms_list = []
            if num_channels == 1 and image.mode != 'L':
                transforms_list.append(torchvision.transforms.Grayscale(num_output_channels=1))
            elif num_channels == 3 and image.mode == 'L':
                 transforms_list.append(torchvision.transforms.Grayscale(num_output_channels=3)) # PIL Grayscale(3) repeats L channel
            elif image.mode == 'RGBA' and num_channels == 3:
                 image = image.convert('RGB')


            transforms_list.extend([
                torchvision.transforms.Resize(target_size),
                torchvision.transforms.ToTensor()
            ])
            # Add normalization if your model expects it, e.g.,
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for 3 channels
            # torchvision.transforms.Normalize(mean=[0.5], std=[0.5]) # for 1 channel
            preprocess = torchvision.transforms.Compose(transforms_list)
            return preprocess(image).unsqueeze(0).to(self.device)
        else: # TensorFlow
            if num_channels == 1:
                img = image.convert('L').resize(target_size)
            else: # num_channels == 3
                img = image.convert('RGB').resize(target_size)

            img_array = tf.keras.preprocessing.image.img_to_array(img) # (H, W, C) or (H,W,1)
            img_array = img_array / 255.0 # Common scaling to [0,1]
            return tf.expand_dims(img_array, 0) # (1, H, W, C)

    def get_prediction(self, image: Image.Image, actual_num_input_channels: int) -> Tuple[Union[torch.Tensor, tf.Tensor], int, float]:
        """Get model prediction for image"""
        input_tensor = self.preprocess_image(image, num_channels=actual_num_input_channels)

        if self.framework == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class].item() * 100
        else:
            output = self.model(input_tensor, training=False) # Ensure inference mode for TF
            probabilities = tf.nn.softmax(output)
            predicted_class = tf.argmax(output[0]).numpy()
            confidence = float(probabilities[0, predicted_class]) * 100

        return input_tensor, predicted_class, confidence

    def integrated_gradients(self, input_tensor, target_class: int, steps: int = 50):
        """Compute Integrated Gradients attribution"""
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
                predictions = self.model(input_tensor)
                if target_class >= predictions.shape[1]:
                    st.error(f"Target class {target_class} out of range for model output with {predictions.shape[1]} classes.")
                    return tf.zeros_like(input_tensor), None
                loss = predictions[:, target_class]

            gradients = tape.gradient(loss, input_tensor)
            if gradients is None:
                st.error("Failed to compute gradients for TensorFlow model (Gradient * Input). Ensure input_tensor is involved in loss calculation.")
                return tf.zeros_like(input_tensor), None
            attributions = gradients * input_tensor
            return attributions, None

    def occlusion(self, input_tensor, target_class: int, window_size: int = 5, stride: int = 3):
        """Compute Occlusion attribution"""
        if self.framework == 'pytorch':
            occlusion_attr = Occlusion(self.model)
            num_channels = input_tensor.shape[1]
            attributions = occlusion_attr.attribute(input_tensor,
                                            target=target_class,
                                            sliding_window_shapes=(num_channels, window_size, window_size),
                                            strides=(1, stride, stride),
                                            baselines=0)
            return attributions
        else: # TensorFlow
            original_output = self.model(input_tensor)
            if target_class >= original_output.shape[1]:
                st.error(f"Target class {target_class} out of range for model output with {original_output.shape[1]} classes.")
                return tf.zeros_like(input_tensor)
            original_score = original_output[0, target_class]

            height, width, num_channels = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
            attributions_np = np.zeros(input_tensor.shape, dtype=np.float32)

            for h in range(0, height - window_size + 1, stride):
                for w in range(0, width - window_size + 1, stride):
                    occluded_np = input_tensor.numpy().copy()
                    occluded_np[0, h:h+window_size, w:w+window_size, :] = 0
                    occluded_tensor = tf.convert_to_tensor(occluded_np)
                    occluded_output = self.model(occluded_tensor)
                    score_drop = original_score - occluded_output[0, target_class]
                    attributions_np[0, h:h+window_size, w:w+window_size, :] += score_drop.numpy()

            if stride < window_size:
                overlap_counts = np.zeros_like(attributions_np)
                for h_idx in range(0, height - window_size + 1, stride):
                    for w_idx in range(0, width - window_size + 1, stride):
                        overlap_counts[0, h_idx:h_idx+window_size, w_idx:w_idx+window_size, :] +=1
                attributions_np = np.divide(attributions_np, overlap_counts, where=overlap_counts!=0, out=np.zeros_like(attributions_np))
            return tf.convert_to_tensor(attributions_np)


    def grad_cam(self, input_tensor, target_class: int, layer_name: str):
        """Compute Grad-CAM attribution"""
        if self.framework == 'pytorch':
            try:
                target_layer_module = dict([*self.model.named_modules()])[layer_name]
            except KeyError:
                st.error(f"Layer '{layer_name}' not found in PyTorch model. Available: {[n for n, _ in self.model.named_modules() if n]}")
                return None
            grad_cam_attr = LayerGradCam(self.model, target_layer_module)
            attributions = grad_cam_attr.attribute(input_tensor, target=target_class)
            if attributions.ndim == 4 and attributions.shape[0] == 1: # Expected (1, C, H, W)
                 # Upsample to original image H, W (from input_tensor)
                attributions = F.interpolate(attributions, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
            return attributions
        else: # TensorFlow
            try:
                conv_layer = self.model.get_layer(layer_name)
            except ValueError:
                st.error(f"Layer '{layer_name}' not found in TensorFlow model. Available: {[l.name for l in self.model.layers]}")
                return None

            grad_model = tf.keras.Model(
                inputs=[self.model.inputs],
                outputs=[conv_layer.output, self.model.output]
            )

            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(input_tensor)
                if target_class >= predictions.shape[1]:
                    st.error(f"Target class {target_class} out of range for model output with {predictions.shape[1]} classes.")
                    return tf.zeros_like(input_tensor)
                loss = predictions[:, target_class]

            grads = tape.gradient(loss, conv_output)
            if grads is None:
                st.error("Failed to compute gradients for Grad-CAM (TF). Ensure selected layer is part of gradient path & model is trainable.")
                return tf.zeros((1, input_tensor.shape[1], input_tensor.shape[2], 1), dtype=tf.float32)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_output_processed = conv_output[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(conv_output_processed)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
            heatmap_resized = tf.image.resize(tf.expand_dims(tf.expand_dims(heatmap,0),-1), [input_tensor.shape[1], input_tensor.shape[2]])
            return heatmap_resized

def visualize_attribution(attributions, title: str):
    """Helper function to visualize attributions"""
    if attributions is None:
        st.warning(f"No attributions to visualize for {title}.")
        return None

    if isinstance(attributions, torch.Tensor):
        attr_np = attributions.squeeze(0).cpu().detach().numpy()
        if attr_np.ndim == 3: # (C, H, W)
            attr_np = np.mean(attr_np, axis=0)
    else: # TensorFlow Tensor
        attr_np = attributions.numpy().squeeze(0)
        if attr_np.ndim == 3 and attr_np.shape[-1] != 1 : # (H, W, C) and C > 1
            attr_np = np.mean(attr_np, axis=-1)
        elif attr_np.ndim == 3 and attr_np.shape[-1] == 1: # (H,W,1)
             attr_np = np.squeeze(attr_np, axis=-1)


    attr_np = np.squeeze(attr_np) # Ensure it's 2D
    if attr_np.ndim != 2:
        st.error(f"Attribution map could not be reduced to 2D for visualization. Final shape: {attr_np.shape}")
        return None

    min_val, max_val = np.min(attr_np), np.max(attr_np)
    if max_val - min_val < 1e-9:
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

    with col1:
        st.header("⚙️ Configuration")
        framework = st.selectbox("Select your framework:", ("pytorch", "tensorflow"), key="framework_select")

        # --- Model Definition Parameters ---
        st.subheader("Model Parameters (for `__init__`)")
        st.markdown("Specify parameters your model class's `__init__` method expects. The example models use these.")

        # Determine actual number of channels for preprocessing based on framework
        actual_num_input_channels_for_preprocessing = 1 if framework == "pytorch" else 3
        
        # User input for `in_channels` if their model definition needs it
        model_def_in_channels = st.number_input(
            f"Number of Input Channels (for model `__init__`):",
            min_value=1, step=1, value=actual_num_input_channels_for_preprocessing,
            help=f"Value to pass as `in_channels` to your model's constructor, if it accepts it. "
                 f"Note: PyTorch images will be preprocessed to {actual_num_input_channels_for_preprocessing} channel(s). "
                 f"TensorFlow images will be preprocessed to {actual_num_input_channels_for_preprocessing} channels."
        )
        if model_def_in_channels != actual_num_input_channels_for_preprocessing:
            st.warning(
                f"Your model `__init__` is set to expect `{model_def_in_channels}` input channels, "
                f"but for **{framework}**, images will be preprocessed to "
                f"`{actual_num_input_channels_for_preprocessing}` channel(s). "
                "Ensure your model definition or preprocessing step handles this correctly."
            )


        # User input for `num_classes`
        num_classes_for_model = st.number_input(
            "Number of Output Classes (for model `__init__` and weights):",
            min_value=1, step=1, value=10,  # Default to 10, user MUST change if different
            help="Crucial! This must match the number of output classes your model was trained on and your weights file corresponds to."
        )


        st.subheader("Step 1: Define Your Model")
        target_img_h, target_img_w = 48, 48 # Assuming this is fixed for preprocessing for now
        
        # Calculate flattened size for default PyTorch model example
        # After 2 MaxPool2d layers with kernel_size=2, stride=2: H_out = H_in/4, W_out = W_in/4
        pytorch_fc_input_size = 32 * (target_img_h // 4) * (target_img_w // 4)

        if framework == 'pytorch':
            default_code = f"""class CustomModel(nn.Module):
    def __init__(self, in_channels={model_def_in_channels}, num_classes={num_classes_for_model}):
        super(CustomModel, self).__init__()
        # Images are preprocessed to {target_img_h}x{target_img_w}
        # And {actual_num_input_channels_for_preprocessing} channel(s) for PyTorch
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # {target_img_h//2}x{target_img_w//2}
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # {target_img_h//4}x{target_img_w//4}
        
        # Flattened size: 32 channels * ({target_img_h//4}) * ({target_img_w//4}) = {pytorch_fc_input_size}
        self.fc1 = nn.Linear({pytorch_fc_input_size}, 128) 
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Use num_classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
"""
        else: # tensorflow
            default_code = f"""class CustomModel(tf.keras.Model):
    def __init__(self, num_classes={num_classes_for_model}, in_channels_tf={model_def_in_channels}):
        super(CustomModel, self).__init__()
        # TF images preprocessed to {target_img_h}x{target_img_w}
        # and {actual_num_input_channels_for_preprocessing} channels (typically 3 for RGB)
        # `input_shape` should match preprocessed image: (H, W, C)
        self.conv1 = layers.Conv2D(16, 3, padding='same', activation='relu', 
                                   input_shape=({target_img_h}, {target_img_w}, in_channels_tf))
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(num_classes) # Use num_classes. Softmax often in loss/outside.

    def call(self, x, training=False): # Add training flag
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
"""

        model_code = st.text_area("Define your model class:", default_code, height=400, key="model_code_area")

        st.subheader("Step 2: Upload Weights")
        weights_file = st.file_uploader(
            "Upload weights file (.pth for PyTorch, .h5 for TensorFlow)",
            type=["pth", "h5"], key="weights_uploader"
        )

        st.subheader("Step 3: Upload Image")
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

    # --- Right Column for Results ---
    with col2:
        st.header("🔍 Interpretation Results")

        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'interpreter' not in st.session_state:
            st.session_state.interpreter = None
        if 'model_instance' not in st.session_state:
            st.session_state.model_instance = None

        # Attempt to load model only if essential components are present and not already loaded
        if model_code and weights_file and not st.session_state.model_loaded:
            with st.spinner("Loading model and weights..."):
                model_class = ModelLoader.load_model_from_code(model_code, framework)
                if model_class:
                    try:
                        init_params = {}
                        sig = inspect.signature(model_class.__init__)

                        if 'in_channels' in sig.parameters:
                            init_params['in_channels'] = model_def_in_channels
                        elif 'in_channels_tf' in sig.parameters: # For TF example
                            init_params['in_channels_tf'] = model_def_in_channels


                        if 'num_classes' in sig.parameters:
                            init_params['num_classes'] = num_classes_for_model
                        
                        st.write(f"Attempting to initialize model with: `{init_params}`") # Debug info

                        model_instance = model_class(**init_params)
                        st.session_state.tmp_model_instance_before_weights = model_instance # For debugging

                        # For TF, build the model if it has a 'build' method and hasn't been built
                        # (often done if input_shape not in first layer)
                        if framework == 'tensorflow' and not model_instance.built and hasattr(model_instance, 'build'):
                            # Determine C for preprocessing
                            c_build = actual_num_input_channels_for_preprocessing
                            dummy_input_shape = (None, target_img_h, target_img_w, c_build)
                            try:
                                st.write(f"Attempting to build TensorFlow model with input shape: {dummy_input_shape}")
                                model_instance.build(dummy_input_shape)
                                st.write("TensorFlow model built.")
                            except Exception as e:
                                st.info(f"Note: TensorFlow model build attempt with {dummy_input_shape} resulted in: {e}. "
                                        "This might be okay if model builds on first call or `input_shape` is in a layer.")


                        model_instance = ModelLoader.load_weights(model_instance, weights_file, framework)

                        if model_instance:
                            st.session_state.model_instance = model_instance
                            st.session_state.interpreter = ModelInterpreter(model_instance, framework)
                            st.session_state.model_loaded = True
                            st.success("Model and weights loaded successfully!")
                        else:
                            st.error("Failed to load weights into the model. See error details above.")
                            st.session_state.model_loaded = False
                    except Exception as e:
                        st.error(f"Error instantiating model or preparing for weights: {str(e)}")
                        st.info(
                            "Double-check your model definition in the text area. "
                            "Ensure `__init__` parameters (like `in_channels`, `num_classes`) match what you've "
                            "entered in the 'Model Parameters' section and that your model code uses them correctly. "
                            "The number of classes is especially important for the final layer to match your weights."
                        )
                        # import traceback
                        # st.code(traceback.format_exc()) # More detailed trace
                        st.session_state.model_loaded = False
        
        if st.session_state.model_loaded and image_file:
            interpreter = st.session_state.interpreter
            model = st.session_state.model_instance
            try:
                image = Image.open(image_file)
                st.image(image, caption='Uploaded Image.', use_container_width=True)

                # Pass the actual number of channels used in preprocessing to get_prediction
                input_tensor, predicted_class, confidence = interpreter.get_prediction(image, actual_num_input_channels_for_preprocessing)
                st.write(f"**Predicted class: `{predicted_class}` with confidence `{confidence:.2f}%`**")
                if predicted_class >= num_classes_for_model:
                    st.error(f"Prediction Alert: Model predicted class `{predicted_class}`, which is "
                             f"outside the specified `num_classes={num_classes_for_model}`. "
                             "This usually means the `Number of Output Classes` setting is incorrect for your model/weights.")


                st.subheader("Step 4: Choose Interpretation Method")
                method_options = ["Integrated Gradients", "Occlusion"]
                if framework == 'pytorch':
                    if any(isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)) for m in model.modules()):
                        method_options.append("Grad-CAM")
                else:
                    if any(isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv1D, tf.keras.layers.Conv3D)) for l in model.layers):
                        method_options.append("Grad-CAM")
                
                method = st.selectbox("Select interpretation method:", method_options, key="interp_method_select")

                # --- Interpretation Method Logic (Copied from previous, check for any new dependencies) ---
                if method == "Integrated Gradients":
                    steps = st.slider("Integration steps (PyTorch IG / TF Gradient*Input):", 20, 200, 50, key="ig_steps")
                    if st.button("Generate IG Interpretation", key="ig_button"):
                        with st.spinner("Computing Integrated Gradients..."):
                            attributions, delta = interpreter.integrated_gradients(input_tensor, predicted_class, steps)
                            if attributions is not None:
                                fig = visualize_attribution(attributions, f"Integrated Gradients (Steps: {steps})")
                                if fig: st.pyplot(fig)

                                st.markdown("---")
                                st.markdown("### Understanding Integrated Gradients (IG)")
                                st.markdown("""
                                IG attributes the model's prediction to its input features (pixels). It calculates pixel importance by integrating gradients along a path from a baseline (e.g., a black image) to the input image.
                                - **What the heatmap shows:** Brighter regions indicate pixels that positively contributed most significantly to the model's prediction for the target class. Darker regions had less or negative impact.
                                - **`Steps` parameter (PyTorch):** Controls the number of steps in the integration. More steps can provide a more accurate attribution but take longer.
                                """)
                                if framework == 'pytorch' and delta is not None:
                                    st.markdown(f"""
                                    - **`Convergence Delta` ({delta.item():.4f}):** This measures the approximation quality of IG. A value closer to 0 indicates that the sum of attributions is close to the difference between the model's output for the input and the baseline.
                                    """)
                                    if abs(delta.item()) > 0.05:
                                        st.warning(f"Convergence Delta ({delta.item():.4f}) is relatively high. Attributions might be less precise. Consider increasing 'steps'.")
                                elif framework == 'tensorflow':
                                    st.markdown("""
                                    - **Note for TensorFlow:** The current TF version uses a simplified **Gradient * Input** attribution, not the full path integral of IG. It shows regions with high gradients scaled by input values. No convergence delta is calculated for this method.
                                    """)

                                with st.expander("How to Interpret the Heatmap & Potential Next Steps"):
                                    st.markdown("""
                                    - **Alignment with Object:**
                                        - **Good sign:** Bright, influential pixels are concentrated on relevant parts of the object (e.g., a cat's face for a "cat" prediction). This suggests the model is focusing on meaningful features.
                                        - **Needs attention:** Bright pixels are scattered, appear in the background, or highlight irrelevant artifacts (e.g., a watermark). This might mean the model is using spurious correlations or biases.

                                    - **Potential Issues & How to Improve:**
                                        1.  **Spurious Correlations:** Model focuses on background or irrelevant details present in training data.
                                            - *Improvement:* Diversify training data, use data augmentation.
                                        2.  **Overfitting to Artifacts:** Model picks up on watermarks, image borders.
                                            - *Improvement:* Clean dataset, augment by randomly adding/removing such artifacts.
                                        3.  **Model Generalization:** Random or nonsensical attributions.
                                            - *Improvement:* More/better data, adjust model complexity, regularization.

                                    - **Actionable Steps:**
                                        - **Analyze Misclassifications:** Use IG on images your model gets wrong.
                                        - **Compare Across Classes:** Do attributions differ meaningfully?
                                        - **Iterate:** Use insights to guide changes in data, training, or model.
                                    """)
                            else:
                                st.error("Failed to compute Integrated Gradients.")


                elif method == "Occlusion":
                    window_size = st.slider("Occlusion window size:", 2, 15, 5, key="occ_window")
                    stride = st.slider("Occlusion stride:", 1, 7, 3, key="occ_stride")
                    if st.button("Generate Occlusion Interpretation", key="occ_button"):
                        with st.spinner("Computing Occlusion... (this can be slow)"):
                            attributions = interpreter.occlusion(input_tensor, predicted_class, window_size, stride)
                            if attributions is not None:
                                fig = visualize_attribution(attributions, f"Occlusion (Window: {window_size}, Stride: {stride})")
                                if fig: st.pyplot(fig)

                                st.markdown("---")
                                st.markdown("### Understanding Occlusion")
                                st.markdown("""
                                Occlusion systematically blocks (occludes) parts of the input image and measures the drop in the model's prediction confidence for the target class.
                                - **What the heatmap shows:** Brighter regions indicate areas where occluding that patch caused the largest drop in confidence.
                                - **`Window Size` / `Stride`:** Control patch size and movement.
                                """)
                                with st.expander("How to Interpret the Heatmap & Potential Next Steps"):
                                    st.markdown("""
                                    - **Alignment with Object:**
                                        - **Good sign:** Occluding parts of the main object leads to a significant drop in confidence.
                                        - **Needs attention:** Occluding background significantly impacts confidence.

                                    - **Potential Issues & How to Improve:**
                                        1.  **Sensitivity to Context:** Model relies heavily on background.
                                            - *Improvement:* Augment data with varied backgrounds.
                                        2.  **Redundant Features:** Occluding one part doesn't change score much, but another similar part does.
                                            - *Improvement:* Check model complexity and feature representation.

                                    - **Actionable Steps:**
                                        - **Experiment with Parameters:** Adjust `window_size` and `stride`.
                                        - **Compare with Other Methods:** Use alongside IG or Grad-CAM.
                                        - **Improve Robustness:** If sensitive to minor occlusions, improve via data augmentation (e.g., cutout).
                                    """)
                            else:
                                st.error("Failed to compute Occlusion attributions.")


                elif method == "Grad-CAM":
                    if framework == 'pytorch':
                        available_layers = [name for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.MaxPool2d)) and name]
                        if not available_layers: available_layers = [name for name, _ in model.named_modules() if name]
                    else: # TensorFlow
                        available_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower() or 'pool' in layer.name.lower()]
                        if not available_layers: available_layers = [layer.name for layer in model.layers]

                    if not available_layers:
                        st.warning("Could not automatically find suitable layers for Grad-CAM.")
                    else:
                        layer_name = st.selectbox("Select layer for Grad-CAM:", available_layers, key="gcam_layer")
                        if st.button("Generate Grad-CAM Interpretation", key="gcam_button"):
                            with st.spinner("Computing Grad-CAM..."):
                                attributions = interpreter.grad_cam(input_tensor, predicted_class, layer_name)
                                if attributions is not None:
                                    fig = visualize_attribution(attributions, f"Grad-CAM: {layer_name}")
                                    if fig: st.pyplot(fig)

                                    st.markdown("---")
                                    st.markdown("### Understanding Grad-CAM")
                                    st.markdown(f"""
                                    Grad-CAM uses gradients flowing into '{layer_name}' to highlight important regions in that layer's feature maps.
                                    - **What the heatmap shows:** Brighter areas indicate regions in the layer's view most influential for the prediction.
                                    - **`Layer Name` ('{layer_name}'):** Early layers show low-level features; later layers show more semantic features.
                                    """)
                                    with st.expander("How to Interpret the Heatmap & Potential Next Steps"):
                                        st.markdown("""
                                        - **Localization & Focus:**
                                            - **Good sign:** Heatmap from a later conv layer highlights the object or discriminative parts.
                                            - **Needs attention:** Heatmap is diffuse, focuses on background.

                                        - **Understanding Layer Behavior:** Examine different layers to see feature evolution.

                                        - **Potential Issues & How to Improve:**
                                            1.  **Poor Localization:** Diffuse heatmaps.
                                                - *Improvement:* Revisit architecture, train longer/better data.
                                            2.  **Bias in Intermediate Features:** Layer focuses on biased features.
                                                - *Improvement:* Data diversification and augmentation.

                                        - **Actionable Steps:**
                                            - **Explore Different Layers.**
                                            - **Compare Across Classes.**
                                            - **Correlate with Performance.**
                                        """)
                                else:
                                    st.error(f"Failed to compute Grad-CAM for layer {layer_name}.")
            except Exception as e:
                st.error(f"An error occurred during image processing or interpretation: {str(e)}")
                import traceback
                st.code(traceback.format_exc()) # For more detailed debugging

        elif st.session_state.model_loaded and not image_file:
            st.info("Model loaded. Please upload an image to proceed with interpretation.")
        elif not model_code or not weights_file:
             st.info("Please define your model and upload weights to begin.")


if __name__ == "__main__":
    main()