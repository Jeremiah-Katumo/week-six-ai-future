import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from PIL import Image
import tensorflow as tf
import io
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="RecycleNet Waste Classifier", layout="wide", page_icon="♻️")

# Constants
REAL_LABELS = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
SIM_LABELS = ["Plastic", "Organic", "Glass", "Metal", "Paper", "E-Waste"]

# Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("./saved_models/recyclenet_model.keras")
        return model
    except Exception:
        st.warning("⚠️ RecycleNet model not found. Falling back to simulation.")
        return None

model = load_model()

# Simulated Prediction
def predict_simulated(image, model_name="ResNet", threshold=0.5):
    probs = np.random.dirichlet(np.ones(len(SIM_LABELS)), size=1)[0]
    results = [(cls, float(prob)) for cls, prob in zip(SIM_LABELS, probs) if prob > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Real Prediction
def predict_real(image, model, threshold=0.0):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    probs = model.predict(img_array)[0]
    results = [(REAL_LABELS[i], float(p)) for i, p in enumerate(probs) if p > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results, probs

# Sidebar Settings
st.sidebar.title("⚙️ Classifier Settings")
model_choice = st.sidebar.selectbox("Model", ["RecycleNet (MobileNetV2)", "ResNet (Simulated)", "EfficientNet (Simulated)"])
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)
image_source = st.sidebar.radio("Image Source", ["Upload Image", "Webcam (Simulated)"])
st.sidebar.markdown("---")
st.sidebar.markdown("Built for sustainable AI 💚")

#  Title
st.title("♻️ RecycleNet: Smart Waste Classification App")

img = None
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload a Waste Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    st.info("Simulated webcam feed (requires sample_waste.jpg)")
    if os.path.exists("sample_waste.jpg"):
        img = Image.open("sample_waste.jpg").convert("RGB")
    else:
        st.error("Missing `sample_waste.jpg`. Upload an image instead.")

# Image Display + Prediction
if img:
    st.image(img, caption="Uploaded Image", use_column_width=False, width=350)

    with st.spinner("🔍 Analyzing image..."):
        time.sleep(1)
        if model_choice == "RecycleNet (MobileNetV2)" and model is not None:
            predictions, full_probs = predict_real(img, model, threshold)
            class_names = REAL_LABELS
        else:
            predictions = predict_simulated(img, model_choice, threshold)
            full_probs = [p for _, p in predictions]
            class_names = SIM_LABELS

    # Display Results
    st.subheader("📊 Prediction Results")
    if predictions:
        result_df = pd.DataFrame(predictions, columns=["Class", "Confidence"])
        result_df["Confidence"] = (result_df["Confidence"] * 100).round(2)

        # Plotly Bar Chart
        fig_bar = px.bar(result_df, x="Confidence", y="Class", orientation='h',
                         color="Class", text="Confidence", title="Class Confidence Distribution")
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_layout(yaxis=dict(categoryorder='total ascending'), height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie Chart
        fig_pie = px.pie(result_df, names="Class", values="Confidence", title="Waste Class Breakdown")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Session History
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Model": model_choice,
            "Time": time.strftime("%H:%M:%S"),
            "Top Class": result_df.iloc[0]["Class"],
            "Confidence": f"{result_df.iloc[0]['Confidence']}%"
        })
    else:
        st.warning("No prediction passed the threshold.")

# History Section
if "history" in st.session_state and st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

# 🧬 Model Info
st.markdown("---")
with st.expander("🧠 View RecycleNet Model Summary"):
    if model:
        stringio = io.StringIO()
        model.summary(print_fn=lambda x: stringio.write(x + '\n'))
        summary_string = stringio.getvalue()
        st.text(summary_string)
    else:
        st.info("Simulated models do not include architecture summary.")

# Footer
st.markdown("---")
footer = """
<div style="text-align: center; padding: 10px 0;">
    <hr>
    <p style="font-size: 0.9em;">🧾 Built with <b>Streamlit</b> & <b>TensorFlow</b> • Developed by <a href="https://github.com/Jeremiah-Katumo">Jeremiah Katumo</a> • 2025</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
