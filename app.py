import streamlit as st
from PIL import Image
import joblib
import os
import pandas as pd
import numpy as np
from io import StringIO

# ---------------------------
# Page config & header
# ---------------------------
st.set_page_config(
    page_title="🎵 Song Popularity Dashboard",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>🎵 Song Popularity Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>An end-to-end Data Science project on music analytics</p>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Optional banner image (optional - ensure folder 'images' exists)
# ---------------------------
banner_path = "images/banner.jpg"
if os.path.exists(banner_path):
    image = Image.open(banner_path)
    resized_image = image.resize((1000, 400))
    st.image(resized_image)
else:
    st.info("Banner image not found at 'images/banner.jpg'. (Optional)")

# ---------------------------
# Author & Project description
# ---------------------------
with st.container():
    st.markdown("### 👤 Author")
    st.markdown("""
    - **Name:** Advait Joshi  
    - **Bio:** AI Engineer Intern @DRDO | Research Intern @IIT Kanpur, IIT Patna | Blockchain Developer Intern @Inspiring Wave | SVIT CSE(DS) '2027  
    - **Linkedin:** [Advait Joshi](https:/www.linkedin.com/in/advaitszone)  
    - **Project Goal:** Showcase complete data science workflow from analysis to deployment.
    """)

st.markdown("---")
st.markdown("### 🧩 Problem Statement")
st.markdown("""
As a data scientist at a music streaming company, your task is to analyze the key **musical** and **platform-related** features that influence a song’s popularity in 2023.

We aim to:
- Predict the popularity of a song based on its **audio** and **platform** attributes.
- Provide **actionable insights** to the **Marketing** and **A&R teams**.
- Help with playlist curation, cross-platform promotion, and artist scouting decisions.
""")

st.markdown("---")
st.markdown("### 🔍 Project Workflow")
st.markdown("""
We followed the complete data science lifecycle:

1. **Understanding the Dataset**  
2. **EDA (Exploratory Data Analysis)**  
3. **Model Building**  
4. **Evaluation**  
5. **Deployment**
""")

st.markdown("---")
st.success("Use the sidebar to navigate through insights, model training, predictions, and final takeaways.")

# ---------------------------
# Load model
# ---------------------------
# NOTE: your file in the repo is named `model[1].pkl`
MODEL_PATH = "model[1].pkl"  # change to "model.pkl" if you rename the file

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success(f"Model loaded from `{MODEL_PATH}`")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning(f"Model file not found at `{MODEL_PATH}`.\nIf your model is large, remove it from repo and load from Drive or use Git LFS.")

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Prediction", "Upload & Batch Predict", "Instructions"])

# ---------------------------
# About page
# ---------------------------
if page == "About":
    st.header("Project Summary")
    st.write("This dashboard shows the end-to-end pipeline and allows prediction using the trained model.")

# ---------------------------
# Prediction (single manual input)
# ---------------------------
elif page == "Prediction":
    st.header("Single Prediction (Manual Input)")
    st.markdown("""
    Paste a **comma-separated** list of feature values matching the order you used during training.
    Example: `0.12, 120.0, 0, 1, 0.53`  
    (If you're not sure about feature order, use CSV upload instead.)
    """)
    user_input = st.text_area("Enter comma-separated features", value="")
    if st.button("Predict (manual)"):
        if model is None:
            st.error("No model loaded. Ensure `model[1].pkl` is present or load model at runtime.")
        else:
            try:
                # parse user input
                arr = [float(x.strip()) for x in user_input.split(",") if x.strip() != ""]
                X = np.array(arr).reshape(1, -1)
                preds = model.predict(X)
                st.write("✅ Prediction:", preds[0])
            except Exception as e:
                st.error(f"Could not predict: {e}\nCheck that you supplied the correct number and type of features.")

# ---------------------------
# Upload & Batch Predict (CSV)
# ---------------------------
elif page == "Upload & Batch Predict":
    st.header("Batch Prediction via CSV")
    st.markdown("Upload a CSV where each row is a sample and columns match the training features (no target column required).")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data (first 5 rows):")
            st.dataframe(df.head())
            if model is None:
                st.error("No model loaded. Cannot run predictions.")
            else:
                if st.button("Run predictions on uploaded CSV"):
                    try:
                        preds = model.predict(df.values)
                        out = df.copy()
                        out["prediction"] = preds
                        st.success("Predictions complete.")
                        st.dataframe(out.head())
                        # allow download
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download results CSV", csv, "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}\nMake sure the CSV columns match the features and ordering used for training.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# ---------------------------
# Instructions page
# ---------------------------
elif page == "Instructions":
    st.header("How to use & deploy")
    st.markdown("""
    **Local run**
    1. Create a virtual environment and install requirements:  
       `pip install -r requirements.txt`  
    2. Run app:  
       `streamlit run app.py`
    3. Make sure `model[1].pkl` is in the same folder or update `MODEL_PATH` variable above.
    
    **Rename model (recommended)**  
    - If desired, rename `model[1].pkl` to `model.pkl` and update `MODEL_PATH = "model.pkl"`.

    **If your model is too large for GitHub**  
    - Upload to Google Drive and load at runtime (use `gdown`), or use Git LFS.
    - Example snippet to download from Drive (add before joblib.load):
    ```python
    import gdown
    url = "https://drive.google.com/uc?id=FILE_ID"
    if not os.path.exists("model.pkl"):
        gdown.download(url, "model.pkl", quiet=False)
    model = joblib.load("model.pkl")
    ```

    **Deploy to Streamlit Cloud**  
    1. Push this repo to GitHub.  
    2. Go to https://share.streamlit.io and connect your GitHub repo.  
    3. Provide the branch and path to `app.py`, then deploy.
    """)
