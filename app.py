import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os

# ---------------- Setup -------------------
st.set_page_config(page_title="📄 Research Summarizer + Q&A", layout="centered")

PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

st.title("📄 Research & News Summarizer + Q&A")

# ---------------- File Upload -------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    filepath = os.path.join(PDF_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Saved: {uploaded_file.name}")

# ---------------- File History -------------------
st.subheader("📂 Previously Uploaded Files")

existing_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
if not existing_files:
    st.info("No files uploaded yet.")
else:
    selected_file = st.selectbox("Select a file to analyze:", existing_files)
    selected_path = os.path.join(PDF_DIR, selected_file)

    # ---------------- Load and Process PDF -------------------
    loader = PyPDFLoader(selected_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    texts = [chunk.page_content for chunk in chunks]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.info("Embedding document... (local model)")
    embeddings = embedder.encode(texts)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Load QA/Summary model
    summarizer = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

    # ---------------- Summary Button -------------------
    if st.button("🔍 Generate Summary"):
        summary_prompt = f"Summarize this:\n\n{texts[0]}"
        summary = summarizer(summary_prompt, max_length=200, truncation=True)[0]["generated_text"]
        st.subheader("📌 Summary")
        st.write(summary)

    # ---------------- Q&A Input -------------------
    st.subheader("💬 Ask a question about the PDF")
    user_question = st.text_input("Type your question:")

    if user_question:
        q_embedding = embedder.encode([user_question])
        D, I = index.search(np.array(q_embedding), k=3)
        relevant_chunks = [texts[i] for i in I[0]]
        context = "\n\n".join(relevant_chunks)

        prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
        answer = summarizer(prompt, max_length=150, truncation=True)[0]["generated_text"]

        st.markdown("**Answer:**")
        st.write(answer)