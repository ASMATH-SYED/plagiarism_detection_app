import streamlit as st
import pdfplumber
import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PATHS (VERY IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

# -----------------------------
# LOAD MODELS
# -----------------------------
with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = pd.read_csv(DATASET_PATH)
dataset_texts = dataset["text"].astype(str).tolist()
dataset_vectors = tfidf.transform(dataset_texts)

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -----------------------------
# PLAGIARISM CHECK
# -----------------------------
def check_plagiarism(input_text):
    vector = tfidf.transform([input_text])
    scores = cosine_similarity(vector, dataset_vectors)
    max_score = scores.max()
    index = scores.argmax()
    return round(max_score * 100, 2), dataset_texts[index]

def plagiarism_level(p):
    if p < 30:
        return "🟢 Low Plagiarism"
    elif p < 60:
        return "🟡 Moderate Plagiarism"
    else:
        return "🔴 High Plagiarism"

# -----------------------------
# UI
# -----------------------------
st.set_page_config("Plagiarism Detection System", layout="wide")

st.title("📘 NLP Based Plagiarism Detection System")
st.write("Enter text or upload a PDF to check plagiarism.")

input_text = st.text_area("Enter text here", height=250)

uploaded_file = st.file_uploader("Upload assignment PDF", type=["pdf"])

pdf_text = ""
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("📄 Extracted Text from PDF")
    st.text_area("PDF Content", pdf_text, height=400)

final_text = input_text.strip() if input_text.strip() else pdf_text.strip()

if st.button("Check Plagiarism"):
    if final_text == "":
        st.warning("Please enter text or upload PDF.")
    else:
        vector = tfidf.transform([final_text])
        prediction = classifier.predict(vector)[0]
        label = "AI Generated Text" if prediction == 1 else "Human Written Text"

        percent, similar_text = check_plagiarism(final_text)
        level = plagiarism_level(percent)

        st.subheader("🔍 Prediction")
        st.write(label)

        st.subheader("📊 Plagiarism Percentage")
        st.progress(percent / 100)
        st.write(f"{percent}%")

        st.markdown(f"### {level}")

        st.subheader("📄 Most Similar Content Found")
        st.write(similar_text)

        report = f"""
Prediction : {label}
Plagiarism % : {percent}
Plagiarism Level : {level}

Most Similar Content:
{similar_text}
"""

        with open("plagiarism_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        with open("plagiarism_report.txt", "rb") as f:
            st.download_button(
                "⬇️ Download Plagiarism Report",
                f,
                file_name="plagiarism_report.txt",
                mime="text/plain"
            )
