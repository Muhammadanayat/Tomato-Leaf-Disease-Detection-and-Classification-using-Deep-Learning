import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Class labels and precautions for each disease
class_labels = [
    "Leaf Mold",
    "Yellow Leaf Curl Virus",
    "Bacterial Spot",
    "Septoria Leaf Spot",
    "Healthy",
    "Spider Mites",
    "Early Blight",
    "Target Spot",
    "Late Blight",
    "Mosaic Virus",
]

precautions = {
    "Leaf Mold": {
        "description": "A fungal disease causing yellow spots and moldy patches on leaves.",
        "precautions": [
            "Ensure proper air circulation around the plants to reduce humidity.",
            "Avoid overhead watering to keep leaves dry."
        ],
        "treatment": [
            "Use fungicides, especially copper-based sprays, to control the infection."
        ]
    },
    "Yellow Leaf Curl Virus": {
        "description": "A viral disease causing yellowing, curling, and distortion of leaves, often spread by whiteflies.",
        "precautions": [
            "Regularly monitor and control whitefly populations using insecticides.",
            "Use reflective mulches or yellow sticky traps to deter whiteflies."
        ],
        "treatment": [
            "Plant virus-resistant tomato varieties.",
            "Remove and destroy infected plants to prevent further spread."
        ]
    },
    "Bacterial Spot": {
        "description": "A bacterial infection resulting in small, dark spots on leaves, stems, and fruits.",
        "precautions": [
            "Avoid overhead watering to prevent waterborne spread.",
            "Disinfect tools and equipment to reduce contamination."
        ],
        "treatment": [
            "Apply copper-based bactericides to manage the infection.",
            "Remove and destroy infected plant parts."
        ]
    },
    "Septoria Leaf Spot": {
        "description": "A fungal disease causing circular spots with dark borders on lower leaves.",
        "precautions": [
            "Remove plant debris and weeds around the tomato plants to eliminate sources of the fungus.",
            "Avoid working in the garden when plants are wet to prevent spread."
        ],
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Prune infected leaves to improve air circulation."
        ]
    },
    "Healthy": {
        "description": "Your plant shows no signs of disease.",
        "precautions": [
            "Maintain regular care and monitoring to ensure continued health.",
            "Avoid overwatering or underwatering."
        ],
        "treatment": [
            "No action is required. Your plant is healthy!"
        ]
    },
    "Spider Mites": {
        "description": "A pest infestation causing yellowing, stippling, and webbing on leaves.",
        "precautions": [
            "Increase humidity around plants to discourage spider mites.",
            "Regularly inspect the undersides of leaves for early signs of infestation."
        ],
        "treatment": [
            "Spray neem oil or insecticidal soap on affected areas.",
            "Use a strong stream of water to dislodge mites from the leaves."
        ]
    },
    "Early Blight": {
        "description": "A fungal disease causing concentric rings and yellowing on older leaves.",
        "precautions": [
            "Avoid planting tomatoes in the same spot consecutively to reduce fungal buildup in the soil.",
            "Remove and destroy infected plant material."
        ],
        "treatment": [
            "Apply fungicides such as mancozeb or chlorothalonil.",
            "Mulch around the plants to prevent soil splash."
        ]
    },
    "Target Spot": {
        "description": "A fungal disease causing circular, target-like spots on leaves.",
        "precautions": [
            "Avoid overwatering and ensure proper drainage.",
            "Space plants adequately to improve airflow."
        ],
        "treatment": [
            "Use fungicides containing azoxystrobin or chlorothalonil."
        ]
    },
    "Late Blight": {
        "description": "A severe fungal disease causing water-soaked spots on leaves and stems, which can rapidly destroy crops.",
        "precautions": [
            "Avoid planting tomatoes near potatoes, as both are susceptible.",
            "Remove and destroy infected plants immediately."
        ],
        "treatment": [
            "Apply fungicides such as mancozeb or chlorothalonil.",
            "Use disease-resistant tomato varieties."
        ]
    },
    "Mosaic Virus": {
        "description": "A viral disease causing mottled patterns and deformities on leaves, often transmitted via tools or plant-to-plant contact.",
        "precautions": [
            "Disinfect tools and hands before handling plants.",
            "Remove infected plants to prevent spread."
        ],
        "treatment": [
            "Use virus-resistant tomato varieties.",
            "Remove and destroy infected plants immediately."
        ]
    }
}


# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_TL_Trainable.keras")

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Resize to the model's input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit App
def main():
    st.title("Tomato Leaf Disease Detection")
    st.write(
        """
# Tomato Leaf Disease Detection Web App

Welcome to the **Tomato Leaf Disease Detection Web App**! This application leverages the power of deep learning to accurately detect and classify diseases in tomato leaves, providing actionable insights and treatment recommendations. 

---

## **Purpose**
This application is designed to assist farmers, gardeners, and agricultural enthusiasts in identifying potential diseases affecting their tomato plants. By simply uploading an image of a tomato leaf, the app will:
- **Classify** the leaf as either healthy or affected by one of 9 diseases.
- Provide **precautions and recommended pesticides** for treating the detected disease.

---

## **Model Performance**
The underlying deep learning model has been trained using advanced transfer learning techniques and achieves an overall accuracy of approximately **91%**. Below is a detailed breakdown of the model's performance for each disease category:

| **Disease**                  | **Precision** | **Recall** | **F1 Score** |
|------------------------------|---------------|------------|--------------|
| **Leaf Mold**                | 0.91          | 0.72       | 0.80         |
| **Yellow Leaf Curl Virus**   | 0.97          | 0.97       | 0.97         |
| **Bacterial Spot**           | 0.96          | 0.91       | 0.93         |
| **Septoria Leaf Spot**       | 0.82          | 0.92       | 0.86         |
| **Healthy**                  | 0.83          | 1.00       | 0.91         |
| **Spider Mites**             | 0.88          | 0.72       | 0.79         |
| **Early Blight**             | 0.82          | 0.72       | 0.77         |
| **Target Spot**              | 0.68          | 0.79       | 0.73         |
| **Late Blight**              | 0.94          | 0.79       | 0.85         |
| **Mosaic Virus**             | 0.58          | 0.99       | 0.73         |

### **Key Metrics Explained**
- **Precision**: Measures how many of the predicted positive cases were actually correct.
- **Recall**: Measures how many of the actual positive cases were identified correctly.
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics.

---

## **How to Use**
1. **Upload an Image**: Use the file uploader to select a clear image of the tomato leaf you want to analyze.
2. **Get a Diagnosis**: The app will analyze the uploaded image and provide:
   - The detected disease (or confirm that the plant is healthy).
   - The confidence score for the prediction.
3. **Treatment Recommendations**: If a disease is detected, the app will display:
   - Precautions to prevent further spread.
   - Recommended pesticides or actions to treat the condition.

---

## **Why Use This App?**
- **Accessible and Easy to Use**: No prior expertise required.
- **Accurate Diagnosis**: Based on a robust deep learning model.
- **Comprehensive Recommendations**: Actionable steps for disease management.

---

Start diagnosing your tomato plants today and ensure healthy, disease-free crops!

        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a Tomato Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

        # Display the result
        st.write(f"### Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Show precautions and pesticides if the plant is diseased
        if predicted_class != "Healthy":
            st.write("### Disease Description:")
            st.write(precautions[predicted_class]["description"])
            
            st.write("### Recommended Precautions:")
            for precaution in precautions[predicted_class]["precautions"]:
                st.write(f"- {precaution}")
            
            st.write("### Treatment Options:")
            for treatment in precautions[predicted_class]["treatment"]:
                st.write(f"- {treatment}")
        else:
            st.write("Your plant is healthy! No action is needed.")


# Run the app
if __name__ == "__main__":
    main()
