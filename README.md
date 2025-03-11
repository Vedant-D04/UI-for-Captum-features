# **Model Interpretability Tool (Captum + Streamlit)**  

This is a **Streamlit-based web application** that helps users visualize and interpret deep learning model predictions using **Captum (for PyTorch) and TensorFlow visualization tools**.  

The app allows users to:  
✅ **Upload a trained model** (`.pth` for PyTorch or `.h5` for TensorFlow).  
✅ **Upload an image** to analyze how the model makes predictions.  
✅ **Select an interpretability method** to visualize important features in the input image.  

---

## **Why use this?**  
Understanding how a deep learning model makes predictions is crucial for debugging, improving model accuracy, and ensuring fairness. This tool provides three popular interpretability techniques to highlight which parts of an image influence the model’s decision.  

---

## **Interpretability Methods**  

### 🔍 GradCAM  
- Generates a heatmap over the image, showing which regions influenced the prediction.  
- Works well for CNN-based models.  

### 🔍 Occlusion Sensitivity  
- Blocks out parts of the image and checks how predictions change.  
- Helps identify which parts of the image contribute most to the model’s confidence.  

### 🔍 Integrated Gradients  
- Analyzes how pixel values contribute to the final prediction using gradients.  
- Useful for both CNNs and other architectures.  

---

## **How to Use the App**  

### **1️⃣ Select a Framework**  
Choose either **PyTorch** or **TensorFlow**. This determines how the app processes your uploaded model.  


### **2️⃣ Upload Your Model**  
⚠️Please Note that the model has only one class, otherwise it will have error while loading.
- **For PyTorch:** Upload a `.pth` file containing your trained model.  
- **For TensorFlow:** Upload a `.h5` file containing your trained model.  
<img width="1677" alt="Screenshot 2025-02-21 at 5 18 22 PM" src="https://github.com/user-attachments/assets/3811908b-cd6a-4c62-8166-d6caef99a85a" />

### **3️⃣ Upload an Image**  
- The app will preprocess the image before passing it to the model.  
- You **may need to modify the preprocessing steps** inside `app.py` to match your trained model’s requirements (e.g., resizing, normalization).  
<img width="1677" alt="Screenshot 2025-02-21 at 5 18 33 PM" src="https://github.com/user-attachments/assets/0f30c7b1-013d-40cc-bcd9-ce9b4699a7ba" />

### **4️⃣ Select an Interpretation Method**  
- Choose **GradCAM, Occlusion, or Integrated Gradients** to analyze the model's prediction.

### **5️⃣ View the Results**  
- The app will display the input image with an **overlay showing important regions** based on the chosen interpretation method.  
<img width="1677" alt="Screenshot 2025-02-21 at 5 18 49 PM" src="https://github.com/user-attachments/assets/59583a21-4211-4efd-ac1f-6bb3a2105bb3" />

<img width="1677" alt="Screenshot 2025-02-21 at 5 19 07 PM" src="https://github.com/user-attachments/assets/ee2175c8-9032-4ccd-9bf3-0a5d046c3ce9" />

<img width="1677" alt="Screenshot 2025-02-21 at 5 19 21 PM" src="https://github.com/user-attachments/assets/0dabc4fb-5010-4342-9656-67ada461f3b8" />

---
Important Notes

⚠️ Preprocessing & Prediction Changes:

The app.py file contains preprocessing and prediction functions that may need adjustments based on your trained model.
Ensure that input dimensions, normalization, and output classes match your model’s requirements.

Future Enhancements

✅ Add support for more interpretability techniques (e.g., LIME, SHAP).
✅ Improve UI/UX with interactive explanations.
✅ Provide pre-trained models for testing.
