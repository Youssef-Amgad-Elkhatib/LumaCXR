import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LumaCXR Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM MEDICAL TERMINAL THEME (CSS INJECTION) ---
st.markdown("""
    <style>
        /* Darken the sidebar and add a cyan border */
        [data-testid="stSidebar"] {
            background-color: #070a13;
            border-right: 1px solid #00e5ff;
        }

        /* Add a cyan neon glow to the main headers */
        h1, h2, h3 {
            color: #00e5ff !important;
            text-shadow: 0px 0px 15px rgba(0, 229, 255, 0.3);
        }

        /* Style the metrics to look like a digital dashboard */
        [data-testid="stMetricValue"] {
            color: #00e5ff;
            font-size: 40px;
        }

        /* Style the file uploader drop-zone */
        [data-testid="stFileUploadDropzone"] {
            border: 2px dashed #00e5ff;
            background-color: #0c1222;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🫁 LumaCXR Hub")
    st.markdown("---")
    st.markdown("**Powered by:** Custom Convolutional Neural Network (CNN)")
    st.markdown("**Target:** Pneumonia")
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This AI is an assistive tool for educational purposes only. It does not replace professional medical diagnosis.")

# --- 3. MAIN HEADER ---
st.title("🫁 LumaCXR: Chest X-Ray Analysis Terminal")
st.markdown("Upload a frontal chest X-ray to initiate the automated neural network scan.")


# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('pneumonia_cnn.h5')


model = load_model()

# --- 5. THE UPLOADER ---
uploaded_file = st.file_uploader("Insert Patient X-Ray (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Use columns to create a side-by-side dashboard UI
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Patient Scan")
        # Added a clean divider for better UI flow
        st.divider()
        st.image(image, use_container_width=True, channels="RGB")

    with col2:
        st.subheader("AI Diagnostics")
        st.divider()

        # --- SIMULATED SCANNING ---
        progress_text = "Initializing neural pathways..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.05)  # Speed of the progress bar

            # Update the text variable at specific milestones
            if percent_complete == 30:
                progress_text = "Extracting spatial features..."
            elif percent_complete == 60:
                progress_text = "Running LumaCXR Custom Architecture..."
            elif percent_complete == 90:
                progress_text = "Compiling diagnostic report..."

            # Update the bar EVERY loop using the current text
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Optional: Pause for half a second at 100% so the user can read the final text
        time.sleep(0.5)
        my_bar.empty()

        # --- PREPROCESSING ---
        img_resized = image.resize((150, 150))
        img_array = np.array(img_resized.convert('L'))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        # --- PREDICTION ---
        prediction = model.predict(img_array)
        confidence = prediction[0][0]

        st.subheader("Scan Complete")

        # --- DYNAMIC RESULTS UI (UPGRADED TO METRICS) ---
        if confidence > 0.5:
            st.error("🚨 **POSITIVE FOR PNEUMONIA**")

            # Interactive metric dashboard
            m1, m2 = st.columns(2)
            m1.metric(label="Primary Diagnosis", value="Pneumonia", delta="Abnormal Scan", delta_color="inverse")
            m2.metric(label="AI Confidence", value=f"{confidence:.2%}")

            st.progress(float(confidence))
            st.info(
                "Visual anomalies detected consistent with focal consolidation or fluid buildup. Recommend immediate radiological review.")
        else:
            normal_conf = 1 - confidence
            st.success("✅ **LUNGS APPEAR NORMAL**")

            # Interactive metric dashboard
            m1, m2 = st.columns(2)
            m1.metric(label="Primary Diagnosis", value="Clear", delta="Normal Scan", delta_color="normal")
            m2.metric(label="AI Confidence", value=f"{normal_conf:.2%}")

            st.progress(float(normal_conf))
            st.info("No significant opacities detected. Lung fields appear clear.")