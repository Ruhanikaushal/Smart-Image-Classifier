import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
from classifier import InferenceEngine

st.set_page_config(
    page_title="Smart Plant Disease Classifier | GDG-USAR",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b6354;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 10px;
    }
    .prediction-card {
        background-color: #e8f5e9;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #c8e6c9;
    }
    .warning-card {
        background-color: #fff8e1;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ffa000;
    }
</style>
""", unsafe_allow_html=True)

# Disease knowledge base with management recommendations
DISEASE_INFO = {
    'Apple scab': {
        'type': 'Fungal (Venturia inaequalis)',
        'advice': 'Apply captan or sulfur-based protectant fungicides early in spring. Prune fallen infected leaves to reduce overwintering spore inocula.'
    },
    'Black rot': {
        'type': 'Fungal (Botryosphaeria obtusa / Guignardia bidwellii)',
        'advice': 'Prune mummified fruit, dead wood, and cankers. Ensure good canopy airflow and apply appropriate labeled fungicides.'
    },
    'Cedar apple rust': {
        'type': 'Fungal (Gymnosporangium juniperi-virginianae)',
        'advice': 'Remove nearby Eastern red cedar host plants within a 1-mile radius where feasible; apply preventive myclobutanil fungicide.'
    },
    'Powdery mildew': {
        'type': 'Fungal (Podosphaera / Erysiphe spp.)',
        'advice': 'Spray potassium bicarbonate or neem oil. Ensure adequate spacing between plants to maximize sunlight and ventilation.'
    },
    'Common rust': {
        'type': 'Fungal (Puccinia sorghi)',
        'advice': 'Plant rust-resistant hybrids; apply strobilurin or triazole fungicides if pustules appear before tassel emergence.'
    },
    'Northern Leaf Blight': {
        'type': 'Fungal (Exserohilum turcicum)',
        'advice': 'Rotate crops out of corn for at least one year; till crop residue into the soil to hasten fungal breakdown.'
    },
    'Early blight': {
        'type': 'Fungal (Alternaria solani)',
        'advice': 'Avoid overhead irrigation to keep foliage dry. Apply copper or chlorothalonil fungicides; mulch the base to block splash-borne spores.'
    },
    'Late blight': {
        'type': 'Oomycete (Phytophthora infestans)',
        'advice': 'High danger disease! Immediately remove and destroy infected plant foliage. Apply preventative copper sprays; do not compost.'
    },
    'Bacterial spot': {
        'type': 'Bacterial (Xanthomonas spp.)',
        'advice': 'Avoid working with wet plants; use pathogen-free seed and copper-mancozeb bactericide combinations.'
    },
    'Target Spot': {
        'type': 'Fungal (Corynespora cassiicola)',
        'advice': 'Maintain good crop canopy air movement. Remove lower diseased leaves and apply approved protectant fungicides.'
    },
    'Septoria leaf spot': {
        'type': 'Fungal (Septoria lycopersici)',
        'advice': 'Remove lower infected leaves, apply drip irrigation, and use preventive fungicides starting at fruit set.'
    },
    'healthy': {
        'type': 'Healthy Plant',
        'advice': 'No disease symptoms detected! Maintain balanced nitrogen-phosphorus fertilization, good soil aeration, and optimal watering.'
    }
}

@st.cache_resource
def load_engine():
    return InferenceEngine(model_path='best_plant_model.pth', classes_path='classes.json')

engine = load_engine()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Google_Developer_Group_logo.svg/800px-Google_Developer_Group_logo.svg.png", width=180)
st.sidebar.markdown("### **Smart Image Classifier**")
st.sidebar.markdown("**GDG-USAR Technical Selection**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Specs:**")
st.sidebar.markdown("- **Backbone:** EfficientNet-B0 (Pretrained)")
st.sidebar.markdown("- **Head:** 3-Layer Custom MLP + Dropout")
st.sidebar.markdown("- **Validation Accuracy:** **99.64%**")
st.sidebar.markdown("- **Trained Classes:** 38 Agricultural Conditions")
st.sidebar.markdown("- **Framework:** PyTorch & Torchvision")

tab1, tab2, tab3 = st.tabs(["🌱 Live Diagnosis & Testing", "📊 Performance & Confusion Matrix", "🔬 Error Analysis & Failure Modes"])

# ================= TAB 1 =================
with tab1:
    st.markdown('<div class="main-header">Smart Plant Disease Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a plant leaf image or test with curated unseen evaluation samples</div>', unsafe_allow_html=True)

    col_input, col_pred = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### 1. Select or Upload Image")
        source_mode = st.radio("Choose Input Method:", ["Sample Test Images", "Upload Custom Image"], horizontal=True)

        selected_image = None
        image_name = ""

        if source_mode == "Sample Test Images":
            test_dir = "test_images"
            if os.path.exists(test_dir):
                sample_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                selected_file = st.selectbox("Choose a preloaded test image:", sample_files)
                if selected_file:
                    selected_image = Image.open(os.path.join(test_dir, selected_file))
                    image_name = selected_file
            else:
                st.warning("test_images directory not found.")
        else:
            uploaded_file = st.file_uploader("Upload leaf photograph (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                image_name = uploaded_file.name

        if selected_image is not None:
            st.image(selected_image, caption=f"Selected Input: {image_name}", use_container_width=True)

    with col_pred:
        st.markdown("#### 2. Model Prediction & Diagnosis")
        if selected_image is not None:
            with st.spinner("Analyzing leaf pathology..."):
                predictions = engine.predict(selected_image, top_k=5)
                top = predictions[0]

            conf = top['confidence_pct']
            badge_color = "#2e7d32" if conf > 90 else ("#f57f17" if conf > 70 else "#c62828")

            st.markdown(f"""
            <div class="prediction-card">
                <span style="font-size:0.9rem; color:#555; text-transform:uppercase; letter-spacing:1px;">Detected Crop</span>
                <h3 style="margin:2px 0 10px 0; color:#1b5e20;">{top['crop'].title()}</h3>
                <span style="font-size:0.9rem; color:#555; text-transform:uppercase; letter-spacing:1px;">Diagnosed Condition</span>
                <h2 style="margin:2px 0 10px 0; color:#2e7d32;">{top['condition'].title()}</h2>
                <div style="background-color:white; border-radius:8px; padding:10px; margin-top:10px;">
                    <div style="display:flex; justify-content:space-between; font-weight:bold; color:#333;">
                        <span>Confidence Score:</span>
                        <span style="color:{badge_color};">{conf:.2f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Disease guidance
            condition_key = next((k for k in DISEASE_INFO if k.lower() in top['condition'].lower()), None)
            if condition_key:
                info = DISEASE_INFO[condition_key]
                st.markdown(f"**Pathogen Type:** `{info['type']}`")
                st.info(f"**Agronomic Guidance:** {info['advice']}")

            # Top 5 bar chart
            st.markdown("##### Top 5 Probabilities:")
            chart_data = pd.DataFrame({
                'Class': [f"{p['crop']} - {p['condition']}" for p in predictions],
                'Confidence (%)': [p['confidence_pct'] for p in predictions]
            }).set_index('Class')
            st.bar_chart(chart_data)
        else:
            st.info("👈 Please select or upload a plant leaf image to begin.")

# ================= TAB 2 =================
with tab2:
    st.markdown("## 📊 Model Evaluation, Training History & Metrics")
    st.markdown("The baseline classifier was built upon an **EfficientNet-B0** convolutional neural network backbone with transfer learning.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Validation Accuracy", "99.64%", "+2.3% over baseline")
    with kpi2:
        st.metric("Best Validation Loss", "0.0148", "Epoch 18")
    with kpi3:
        st.metric("Total Classes", "38 Classes", "14 Crops")
    with kpi4:
        st.metric("Validation Set Size", "17,572 Images", "Augmented Split")

    st.markdown("---")
    st.subheader("Training and Validation Curves")
    curves_path = "Documents/training_curves.png" if os.path.exists("Documents/training_curves.png") else "training_curves.png"
    if os.path.exists(curves_path):
        st.image(curves_path, caption="Loss and Accuracy convergence over 23 epochs with Early Stopping", use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix Analysis")
    cm_focused = "Documents/confusion_matrix_focused.png" if os.path.exists("Documents/confusion_matrix_focused.png") else "confusion_matrix_focused.png"
    cm_full = "Documents/confusion_matrix_full.png" if os.path.exists("Documents/confusion_matrix_full.png") else "confusion_matrix_full.png"
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        if os.path.exists(cm_focused):
            st.image(cm_focused, caption="Focused Confusion Matrix: Key Tomato, Potato, Apple & Corn Diseases", use_container_width=True)
    with col_cm2:
        if os.path.exists(cm_full):
            st.image(cm_full, caption="Full 38x38 Class Confusion Matrix", use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Summary")
    st.markdown("""
    - **Macro Average Precision:** 99.65%
    - **Macro Average Recall:** 99.64%
    - **Macro Average F1-Score:** 99.64%
    - **Early Stopping Trigger:** Patience = 5 epochs; automatically halted training at Epoch 23 to prevent overfitting.
    """)

# ================= TAB 3 =================
with tab3:
    st.markdown("## 🔬 Error Analysis & Failure Case Investigation")
    st.markdown("""
    Requirement: *Record one model failure and your likely explanation in DECISIONS.md.*
    Below is a breakdown of our stress-testing, real failure cases, and root-cause explanations.
    """)

    st.markdown("### ⚠️ Documented Failure Case: Illumination Collapse & Optical Blur")
    col_fail1, col_fail2 = st.columns([1, 1])

    with col_fail1:
        if os.path.exists("test_images/failure_case_underexposed_blur.jpg"):
            st.image("test_images/failure_case_underexposed_blur.jpg", caption="Corrupted Input: Underexposed & Blurred Tomato Leaf", width=300)
    with col_fail2:
        st.markdown("""
        <div class="warning-card">
            <h4>Observed Failure Details</h4>
            <ul>
                <li><b>Ground Truth:</b> <code>Tomato___healthy</code></li>
                <li><b>Model Prediction:</b> <code>Corn_(maize)___healthy</code> (Confidence: 67.41%)</li>
                <li><b>Second Candidate:</b> <code>Soybean___healthy</code> (Confidence: 8.33%)</li>
                <li><b>Classification Result:</b> ❌ <b>Misclassified Crop Species</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧠 Likely Root-Cause Explanation")
    st.markdown("""
    1. **Loss of High-Frequency Edge & Serration Features:**
       Healthy tomato leaflets possess distinctive pointed serrated margins and reticulate venation. When optical blur is applied, high-frequency spatial gradients are smoothed out, destroying the fine features used by EfficientNet's depthwise-separable conv layers.
    2. **Low-Frequency Dominance & Shape Confusion:**
       Under severe underexposure and blur, only the primary elongated green contour remains. The network confuses the broad, blurred leaf silhouette with a corn (maize) leaf lamina.
    3. **Closed-World Softmax Distribution:**
       Standard softmax forces logits across known classes to sum to 1.0. Even when an input is severely distorted or out-of-domain, the model cannot output *"Unknown / Degraded Image"*; it is forced to assign probability to the closest training representation.
    """)

    st.markdown("### 🛠️ Production Mitigations & Proposed Fixes")
    st.markdown("""
    - **Input Quality Gate:** Introduce an OpenCV blur detection pre-filter (e.g. Laplacian variance $> 100$) and exposure thresholding before passing frames to the classifier.
    - **Epistemic Uncertainty via MC Dropout:** Enable dropout at inference time (Monte Carlo Dropout) to estimate predictive variance and flag uncertain predictions.
    - **Out-of-Distribution (OOD) Detection:** Add an energy-based OOD score or an explicit 'background/unidentified' class to reject non-leaf or corrupted inputs.
    """)
