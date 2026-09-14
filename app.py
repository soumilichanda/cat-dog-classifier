import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Cat vs Dog: Model Comparison",
    page_icon="🐾",
    layout="wide"
)

@st.cache_resource
def load_all_models():
    """
    Loads custom classification models and the ImageNet-1k pre-trained backbone.
    Cached in RAM to prevent disk I/O latency across Streamlit reruns.
    """
    baseline = tf.keras.models.load_model("models/baseline_cnn.keras")
    mobilenet_custom = tf.keras.models.load_model("models/mobilenetv2_final.keras")
    
    # 1000-class full ImageNet classifier used as a semantic verification gatekeeper
    detector = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    return baseline, mobilenet_custom, detector

with st.spinner("Loading vision engines into memory..."):
    baseline_model, mobilenet_model, object_detector = load_all_models()

st.sidebar.title("📊 Benchmark Evaluation")
st.sidebar.table({
    "Metric": ["Val Accuracy", "False Predictions", "Input Scaling", "Backbone"],
    "Custom CNN": ["~85.0%", "701", "[0, 1]", "3-Block CNN"],
    "MobileNetV2": ["97.4%", "120", "[-1, 1]", "ImageNet"]
})

st.sidebar.divider()
show_cm = st.sidebar.checkbox("Show Confusion Matrix comparison", value=False)
if show_cm:
    cm_path = "models/confusion_matrix_comparison.png"
    if os.path.exists(cm_path):
        st.sidebar.image(
            cm_path, 
            caption="Validation Split Confusion Matrices", 
            use_container_width=True
        )

st.title("🐾 Cat vs Dog: Comparative Vision Engine")
st.caption(
    "A comparative evaluation between a scratch-trained CNN and a fine-tuned "
    "MobileNetV2 architecture with robust non-pet object verification."
)

uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

def verify_if_cat_or_dog(raw_img):
    """
    Robust Semantic Gatekeeper:
    Uses exact ImageNet-1k ground truth index ranges rather than heuristic keyword matching:
      - Indices 151 to 268: All 120 domestic dog breeds (Samoyed, Saluki, Kuvasz, etc.)
      - Indices 281 to 285: Domestic cats (Tabby, Tiger cat, Persian, Siamese, Egyptian)
      - Indices 286 to 292: Wild felids (Cougar, Lynx, Leopard, Lion, Cheetah) are strictly excluded.
    """
    # 1. Standardize image tensor to native ImageNet dimensions (224x224)
    img_224 = raw_img.resize((224, 224))
    arr_224 = np.array(img_224, dtype=np.float32)
    arr_224 = tf.keras.applications.mobilenet_v2.preprocess_input(arr_224)
    batch_224 = np.expand_dims(arr_224, axis=0)

    # 2. Extract raw un-decoded prediction vector across all 1,000 ImageNet categories
    preds = object_detector.predict(batch_224, verbose=0)[0]
    top_5_indices = np.argsort(preds)[-5:][::-1]

    # Ground truth contiguous ImageNet-1k index sets for domestic pets
    DOMESTIC_PET_INDICES = set(range(151, 269)).union(set(range(281, 286)))

    top_idx = top_5_indices[0]
    decoded_top = tf.keras.applications.mobilenet_v2.decode_predictions(
        np.expand_dims(preds, axis=0), top=1
    )[0][0]
    top_detected_name = decoded_top[1].replace("_", " ").title()
    top_detected_prob = float(preds[top_idx]) * 100.0

    # 3. Check if any top-5 candidate belongs to verified domestic pet classes
    for idx in top_5_indices:
        if idx in DOMESTIC_PET_INDICES and preds[idx] > 0.08:
            return True, top_detected_name, top_detected_prob

    # Bypassed: input is an inanimate object, everyday item, or wild predator
    return False, top_detected_name, top_detected_prob

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Uploaded Input", use_container_width=True)

    with col2:
        st.subheader("Model Predictions")

        # Step 1: Execute ImageNet semantic gatekeeper
        is_pet, detected_obj, obj_score = verify_if_cat_or_dog(raw_image)

        if not is_pet:
            st.error("### Result: Other ❓ (Neither Cat nor Dog)")
            st.warning(
                f"**Object Verification Alert:**\n\n"
                f"The system detected: **{detected_obj}** ({obj_score:.1f}% confidence).\n\n"
                f"This input does not belong to domestic feline or canine classes, "
                f"so binary classification was bypassed to prevent false predictions."
            )
        else:
            # Step 2: Format tensor to custom classifier specifications (128x128)
            resized_img = raw_image.resize((128, 128))
            img_array = np.array(resized_img, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)

            # --- 1. Custom Baseline CNN ---
            baseline_raw_prob = float(baseline_model.predict(img_batch, verbose=0)[0][0])
            if baseline_raw_prob >= 0.5:
                base_label = "Dog 🐶"
                base_conf = baseline_raw_prob * 100.0
            else:
                base_label = "Cat 🐱"
                base_conf = (1.0 - baseline_raw_prob) * 100.0

            st.markdown("### 1. Custom Baseline CNN")
            st.write(f"Prediction: **{base_label}** ({base_conf:.2f}% confidence)")
            st.progress(min(max(base_conf / 100.0, 0.0), 1.0))

            st.divider()

            # --- 2. MobileNetV2 (Transfer Learning) ---
            mobilenet_raw_prob = float(mobilenet_model.predict(img_batch, verbose=0)[0][0])
            if mobilenet_raw_prob >= 0.5:
                mb_label = "Dog 🐶"
                mb_conf = mobilenet_raw_prob * 100.0
            else:
                mb_label = "Cat 🐱"
                mb_conf = (1.0 - mobilenet_raw_prob) * 100.0

            st.markdown("### 2. MobileNetV2 (Transfer Learning)")
            st.write(f"Prediction: **{mb_label}** ({mb_conf:.2f}% confidence)")
            st.progress(min(max(mb_conf / 100.0, 0.0), 1.0))

else:
    st.info("Upload any cat, dog, or non-pet image above to test the system live.")