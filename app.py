import streamlit as st
import pdfplumber
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# FILE PATHS (STREAMLIT SAFE)
# ===============================

TFIDF_PATH = "model/tfidf.pkl"
CLASSIFIER_PATH = "model/classifier.pkl"
DATASET_PATH = "data/dataset.csv"

# ===============================
# LOAD MODELS
# ===============================

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

# ===============================
# LOAD DATASET
# ===============================

dataset = pd.read_csv(DATASET_PATH)
dataset_texts = dataset["text"].astype(str).tolist()
dataset_vectors = tfidf.transform(dataset_texts)

# ===============================
# PDF TEXT EXTRACTION
# ===============================

def extract_text_from_pdf(uploaded_file):
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

# ===============================
# PLAGIARISM CHECK
# ===============================

def check_plagiarism(input_text):
    input_vector = tfidf.transform([input_text])
    similarity_scores = cosine_similarity(input_vector, dataset_vectors)

    max_score = similarity_scores.max()
    index = similarity_scores.argmax()

    plagiarism_percent = round(max_score * 100, 2)
    most_similar_text = dataset_texts[index]

    return plagiarism_percent, most_similar_text

# ===============================
# PLAGIARISM LEVEL
# ===============================

def plagiarism_level(percent):
    if percent < 30:
        return "🟢 Low Plagiarism"
    elif percent < 60:
        return "🟡 Moderate Plagiarism"
    else:
        return "🔴 High Plagiarism"

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="Plagiarism Detection System", layout="wide")

st.title("📘 NLP Based Plagiarism Detection System")
st.write("Enter text or upload a PDF to check plagiarism.")

# ===============================
# TEXT INPUT
# ===============================

input_text = st.text_area("Enter text here:", height=250)

# ===============================
# PDF UPLOAD
# ===============================

uploaded_file = st.file_uploader(
    "Upload assignment PDF",
    type=["pdf"]
)

extracted_text = ""

if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📄 Extracted Text from PDF")
    st.text_area("PDF Content", extracted_text, height=400)

# ===============================
# FINAL INPUT SELECTION
# ===============================

final_text = input_text.strip() if input_text.strip() else extracted_text.strip()

# ===============================
# CHECK PLAGIARISM
# ===============================

if st.button("Check Plagiarism"):

    if final_text == "":
        st.warning("Please enter text or upload a PDF.")
    else:

        vector = tfidf.transform([final_text])
        prediction = classifier.predict(vector)[0]

        prediction_label = (
            "AI Generated Text" if prediction == 1 else "Human Written Text"
        )

        plagiarism_percent, similar_text = check_plagiarism(final_text)
        level = plagiarism_level(plagiarism_percent)

        st.subheader("🔍 Prediction")
        st.write(prediction_label)

        st.subheader("📊 Plagiarism Percentage")
        st.progress(plagiarism_percent / 100)
        st.write(f"{plagiarism_percent}%")

        st.markdown(f"### Plagiarism Level: {level}")

        st.subheader("📄 Most Similar Content Found")
        st.write(similar_text)

        # ===============================
        # REPORT GENERATION (FULL TEXT)
        # ===============================

        report_text = f"""
NLP BASED PLAGIARISM DETECTION REPORT
-----------------------------------

Prediction        : {prediction_label}
Plagiarism %      : {plagiarism_percent}%
Plagiarism Level  : {level}

Most Similar Content
-----------------------------------
{similar_text}
"""

        with open("plagiarism_report.txt", "w", encoding="utf-8") as file:
            file.write(report_text)

        with open("plagiarism_report.txt", "rb") as file:
            st.download_button(
                label="⬇️ Download Plagiarism Report",
                data=file,
                file_name="plagiarism_report.txt",
                mime="text/plain"
            )