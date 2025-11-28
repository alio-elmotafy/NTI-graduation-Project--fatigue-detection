import streamlit as st
import time
import os

# ---------- 1. CONFIGURATION (MUST BE FIRST) ----------
st.set_page_config(
    page_title="Driver Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Suppress annoying TensorFlow/OneDNN warnings in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------- 2. ROBUST IMPORTS (Prevent Version Crashes) ----------
try:
    import numpy as np
    import joblib
    from PIL import Image
    
    # Robust TensorFlow Import
    try:
        import tensorflow as tf
        # Try accessing Keras the robust way for TF 2.x
        try:
            MobileNetV2 = tf.keras.applications.MobileNetV2
            preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
            img_to_array = tf.keras.preprocessing.image.img_to_array
            load_img = tf.keras.preprocessing.image.load_img
        except AttributeError:
            # Fallback for older/mixed environments
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from tensorflow.keras.preprocessing.image import img_to_array
            from tensorflow.keras.preprocessing.image import load_img

    except ImportError as tf_error:
        st.error(f"TensorFlow Import Error: {tf_error}")
        st.stop()

except ValueError as e:
    # Handle the Numpy/TF version conflict gracefully
    if "numpy" in str(e).lower() or "binary incompatibility" in str(e).lower():
        st.error("⚠️ LIBRARY VERSION CONFLICT DETECTED")
        st.markdown("Please run: `pip install \"numpy==1.24.3\"`")
        st.stop()
    raise e

# ---------- 3. CUSTOM CSS (Your New Theme) ----------
st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700&display=swap');

    /* Main App Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 15, 25) 0%, rgb(5, 5, 10) 90%);
        font-family: 'Exo 2', sans-serif;
    }

    /* Headings & Text */
    h1, h2, h3, h4, h5, h6, p, div {
        font-family: 'Exo 2', sans-serif !important;
        color: #e0e0e0;
    }

    /* Gradient Title */
    h1 {
        text-transform: uppercase;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 2px;
        text-shadow: 0px 0px 30px rgba(0, 242, 254, 0.3);
        margin-bottom: 30px;
    }

    /* Custom File Uploader */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #4facfe;
        background-color: rgba(79, 172, 254, 0.1);
    }

    /* Modern Glass Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #000;
        border: none;
        padding: 12px 30px;
        border-radius: 30px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5);
    }

    /* Metric Cards Glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
    }
    div[data-testid="stMetricLabel"] {
        color: #888;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #fff;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- 4. HEADER ----------
st.title("Driver Guard AI")
st.markdown("##### Intelligent Safety Monitoring System")

# ---------- 5. LOAD MODELS (Targeted Loading) ----------
@st.cache_resource
def load_models_safe():
    try:
        # CNN Feature Extractor (MobileNetV2)
        base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        
        # Load Classifier (Strictly looking for the Voting Classifier pickle)
        model_path = 'drowsiness_detection_model.pkl'
        
        if os.path.exists(model_path):
            main_model = joblib.load(model_path)
            return base_cnn, main_model, "SUCCESS"
        else:
            return None, None, "MISSING_FILE"

    except Exception as e:
        return None, None, str(e)

# Load the system
cnn_model, main_model, status = load_models_safe()

# Check Status
if status == "MISSING_FILE":
    st.error("❌ Model file missing!")
    st.warning("Please upload 'drowsiness_detection_model.pkl' to the directory.")
    st.stop()
elif status != "SUCCESS":
    st.error(f"❌ Error loading model: {status}")
    st.stop()

# ---------- 6. IMAGE PROCESSING ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📤 Upload Feed")
    uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

def preprocess_image(pil_img, use_bgr_flag=False):
    # Resize
    img = pil_img.resize((224, 224))
    # To Array
    x = img_to_array(img)
    
    # BGR conversion if requested
    if use_bgr_flag:
        x = x[..., ::-1]
        
    # Expand dims
    x = np.expand_dims(x, axis=0)
    # Preprocess (MobileNetV2 scaling)
    x = preprocess_input(x)
    return x

# ---------- 7. SETTINGS EXPANDER (Hidden by default) ----------
# Defaults
invert_prediction = False
use_bgr = False

# ---------- 8. PREVIEW & PREDICTION ----------
if uploaded_file is not None:
    # Open image with PIL to be robust
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    with col2:
        st.markdown("#### 👁️ Preview")
        st.image(image_pil, use_column_width=True)

    st.markdown("---")
    
    # Center the button
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze_btn = st.button('SCAN DRIVER STATUS')

    if analyze_btn:
        input_tensor = preprocess_image(image_pil, use_bgr)
        
        with st.spinner('Analyzing biometric features...'):
            # CNN features extraction
            features = cnn_model.predict(input_tensor)  # shape: (1, 1280)
            
            # --- PREDICTION LOGIC (Voting Classifier) ---
            # Main Hard Vote Prediction (0 or 1)
            prediction = main_model.predict(features)[0]
            
            # Logic Inversion Handling
            # Assuming 0 = Fatigue, 1 = NonFatigue (Alert) based on typical datasets
            is_fatigue = (prediction == 0)
            
            if invert_prediction:
                is_fatigue = not is_fatigue
                
            final_status = "Fatigue" if is_fatigue else "NonFatigue"
            
            # --- METRICS CALCULATION ---
            # Since it's Hard Voting, we can't get predict_proba directly.
            # However, we can sometimes peek at the individual estimators to show "Votes"
            vote_info = "Hard Vote"
            reliability = "Verified"
            
            try:
                # Attempt to access internal estimators if available
                if hasattr(main_model, 'estimators_'):
                    # Count how many estimators voted for the predicted class
                    votes_for_prediction = 0
                    total_estimators = len(main_model.estimators_)
                    
                    for estimator in main_model.estimators_:
                        try:
                            est_pred = estimator.predict(features)[0]
                            if est_pred == prediction:
                                votes_for_prediction += 1
                        except:
                            pass # Some estimators might fail individually or differ in API
                            
                    if total_estimators > 0:
                        vote_percentage = (votes_for_prediction / total_estimators) * 100
                        vote_info = f"{votes_for_prediction}/{total_estimators} Votes"
                        reliability = f"{vote_percentage:.0f}% Agreement"
            except:
                pass # Fallback if introspection fails

            # ---------- DISPLAY RESULTS ----------
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("System Mode", "Ensemble Voting")
            with m2:
                st.metric("Detected State", final_status)
            with m3:
                st.metric("Model Consensus", vote_info, delta=reliability)

            st.markdown("<br>", unsafe_allow_html=True)

            # Final Banner
            if final_status == "Fatigue":
                st.markdown(
                    """
                    <div style="background: rgba(255, 75, 75, 0.1); 
                                padding: 30px; border-radius: 20px; text-align: center; 
                                border: 1px solid rgba(255, 75, 75, 0.3); backdrop-filter: blur(10px);
                                box-shadow: 0 0 50px rgba(255, 0, 0, 0.2);">
                        <h1 style="color: #ff4b4b; margin:0; font-size: 2.5rem; text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);">⚠️ FATIGUE DETECTED</h1>
                        <p style="color: #ffcdd2; font-size: 1.2rem; margin-top: 10px; font-weight: 300;">
                            Driver appears drowsy. Immediate action required.
                        </p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background: rgba(0, 242, 96, 0.1); 
                                padding: 30px; border-radius: 20px; text-align: center; 
                                border: 1px solid rgba(0, 242, 96, 0.3); backdrop-filter: blur(10px);
                                box-shadow: 0 0 50px rgba(0, 255, 0, 0.2);">
                        <h1 style="color: #00f260; margin:0; font-size: 2.5rem; text-shadow: 0 0 10px rgba(0, 242, 96, 0.5);">✅ DRIVER IS ALERT</h1>
                        <p style="color: #ccffcc; font-size: 1.2rem; margin-top: 10px; font-weight: 300;">
                            Driver is awake and attentive. Status Normal.
                        </p>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.balloons()

# ---------- 9. FOOTER & SETTINGS ----------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")

# Settings in an expander at the bottom
with st.expander("⚙️ Advanced Calibration"):
    c1, c2 = st.columns(2)
    with c1:
        invert_prediction = st.checkbox("Invert Logic", value=False)
    with c2:
        use_bgr = st.checkbox("Use BGR Color", value=False)

# Credits at the bottom - WHITE color
st.markdown(
    """
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <p style="color: white !important; font-weight: bold !important; font-size: 1.2rem; margin: 0; padding-bottom: 5px; opacity: 0.9;">Made by: Ali Osama</p>
        <p style="color: white !important; font-weight: bold !important; font-size: 1.2rem; margin: 0; opacity: 0.9;">Instructor: Dr. Elhossiny Ibrahim</p>
    </div>
    """,
    unsafe_allow_html=True
)