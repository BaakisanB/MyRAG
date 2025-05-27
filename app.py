import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI

# ---------------- Setup -------------------
st.set_page_config(page_title="📄 Research Summarizer + Q&A", layout="centered")

# OpenAI-compatible local API from LM Studio
client = OpenAI(
    base_url = "http://localhost:1234/v1",
    api_key = "lm-studio"
)

PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

st.title("📄 Research & News Summarizer + Q&A (LLaMA 3)")

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

    # ---------------- Summary Button -------------------
    if st.button("🔍 Generate Summary"):
        summary_prompt = f"Summarize the following:\n\n{texts[0]}\n\nSummary:"
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        summary = response.choices[0].message.content
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

        qa_prompt = f"Context:\n{context}\n\nQuestion: {user_question}\n\nAnswer:"
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": qa_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        answer = response.choices[0].message.content
        st.markdown("**Answer:**")
        st.write(answer)