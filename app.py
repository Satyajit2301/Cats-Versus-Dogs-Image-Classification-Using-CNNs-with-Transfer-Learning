import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Premium Look
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f093fb, #f5576c, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #b0b0c0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }

    .pred-cat {
        border-left: 4px solid #f093fb;
    }

    .pred-dog {
        border-left: 4px solid #ffd200;
    }

    .pred-label {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .pred-confidence {
        font-size: 1.3rem;
        color: #b0b0c0;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model file location
# ============================================================
MODEL_FILENAME = "best_model_efficientnetb0.keras"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)


# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained EfficientNetB0 model (cached across reruns).

    Returns None if the model file isn't next to this script, so the
    caller can show a clean error instead of crashing.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    # compile=False: we only need inference here, not the optimizer/loss
    # state, and it avoids unnecessary warnings/slowdowns on load.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


# ============================================================
# Preprocessing function
# ============================================================
def preprocess_image(image):
    """Apply the same preprocessing used during training.

    The model was trained on images that were resized to 128x128 and
    rescaled from [0, 255] to [0, 1] (tf.keras.layers.Rescaling(1./255)).
    The EfficientNet-specific preprocessing (converting back to the
    [0, 255]-style range it expects) and the augmentation layer are
    already baked into the saved model graph itself, so we must NOT
    apply them again here - we only resize and rescale to [0, 1].
    """
    img = image.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ============================================================
# Main App
# ============================================================
st.markdown('<h1 class="main-title">🐱 Cats vs Dogs 🐶</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Image Classification using EfficientNetB0 with Transfer Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 About")
    st.markdown("""
    This app classifies images of **cats** and **dogs** using a
    deep learning model built with **EfficientNetB0** and
    **Transfer Learning**.

    **Model Details:**
    - Architecture: EfficientNetB0
    - Pre-trained on: ImageNet
    - Fine-tuned for: Binary Classification
    - Input Size: 128 x 128 px
    - Test Accuracy: ~89.3%
    - AUC: 0.96
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - TensorFlow / Keras
    - Streamlit
    - Python
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Satyajit** | Lab Assignment 03")

# Load the model up front so we fail fast with a clear message
# instead of crashing later inside the prediction flow.
model = load_model()

if model is None:
    st.error(
        f"⚠️ Model file **{MODEL_FILENAME}** was not found in the same "
        f"folder as `app.py`.\n\n"
        f"Download it from your notebook (the Block 13/14 export step) "
        f"and place it next to this script, then restart the app."
    )
    st.stop()

# File uploader
st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose a cat or dog image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    help="Upload a clear image of a cat or dog for classification"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(image, width="stretch")

        # Show image info
        w, h = image.size
        st.caption(f"Dimensions: {w} x {h} px | Mode: {image.mode}")

    with col2:
        st.markdown("#### 🔮 Prediction")

        with st.spinner("Analyzing image..."):
            try:
                processed = preprocess_image(image)
                prediction = float(model.predict(processed, verbose=0)[0][0])
            except Exception as e:
                st.error(f"Something went wrong while running the model: {e}")
                st.stop()

            # Determine class
            if prediction >= 0.5:
                pred_class = "🐶 Dog"
                confidence = prediction * 100
                css_class = "pred-dog"
                emoji = "🐶"
                color = "#ffd200"
            else:
                pred_class = "🐱 Cat"
                confidence = (1 - prediction) * 100
                css_class = "pred-cat"
                emoji = "🐱"
                color = "#f093fb"

        # Display prediction
        st.markdown(f"""
        <div class="prediction-box {css_class}">
            <div style="font-size: 4rem;">{emoji}</div>
            <div class="pred-label" style="color: {color};">{pred_class}</div>
            <div class="pred-confidence">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("**Confidence Breakdown:**")
        cat_prob = (1 - prediction) * 100
        dog_prob = prediction * 100
        st.progress(float(cat_prob / 100), text=f"🐱 Cat: {cat_prob:.1f}%")
        st.progress(float(dog_prob / 100), text=f"🐶 Dog: {dog_prob:.1f}%")

    # Additional info
    st.markdown("---")
    st.markdown("#### 📊 Prediction Details")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    with detail_col1:
        st.metric("Predicted Class", pred_class.split(" ")[1])
    with detail_col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    with detail_col3:
        st.metric("Raw Score", f"{prediction:.4f}")

else:
    # Placeholder when no image uploaded
    st.markdown("""
    <div class="prediction-box">
        <div style="font-size: 3rem;">📸</div>
        <p style="color: #b0b0c0; font-size: 1.1rem;">
            Upload a cat or dog image to get started!
        </p>
        <p style="color: #777; font-size: 0.9rem;">
            Supported formats: JPG, JPEG, PNG, BMP, WEBP
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using TensorFlow & Streamlit | Cats vs Dogs Classification | Lab Assignment 03</p>
</div>
""", unsafe_allow_html=True)