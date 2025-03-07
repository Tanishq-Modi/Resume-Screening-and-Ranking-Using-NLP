import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
import numpy as np
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to extract and clean text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handles NoneType error
    return clean_text(text)


# Function to clean text (removes special characters, extra spaces, and converts to lowercase)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    return text.lower().strip()  # Convert to lowercase


# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarity_scores = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return similarity_scores


# Streamlit app configuration
st.set_page_config(page_title="AI Resume Screening", layout="wide", page_icon="📄")

st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description.strip():
    st.header("⚡ Processing Resumes...")

    resumes = []
    valid_files = []
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if text:  # Only add resumes that have valid text
            resumes.append(text)
            valid_files.append(file.name)

        progress_bar.progress((i + 1) / len(uploaded_files))  # Update progress

    if resumes:
        # Rank Resumes
        scores = rank_resumes(job_description, resumes)

        # Create results DataFrame
        results = pd.DataFrame({"Resume": valid_files, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        # Display Ranking Results
        st.subheader("🎯 Ranked Resumes")
        st.dataframe(results.style.background_gradient(cmap="Blues"), use_container_width=True)

        # Show progress bars for each resume
        st.subheader("📊 Resume Score Visualization")
        for i, row in results.iterrows():
            st.markdown(f"**{row['Resume']}**")
            st.progress(float(row["Score"]))

        # Generate word cloud for job description
        st.subheader("📢 Job Description Keyword Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(job_description)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Download option for ranked results
        st.subheader("📥 Download Results")
        st.download_button(
            label="Download Ranking as CSV",
            data=results.to_csv(index=False),
            file_name="ranked_resumes.csv",
            mime="text/csv"
        )

        st.success("✅ Resume ranking completed successfully!")
    else:
        st.error("⚠️ No valid resumes found. Please upload files with extractable text.")

else:
    st.warning("⚠️ Please enter a job description and upload resumes to proceed.")
