from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Load PDF
loader = PyPDFLoader("01.pdf")
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
texts = [chunk.page_content for chunk in chunks]

# Embed chunks using a local model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts)

# Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Ask a question
question = input("Ask a question: ")
q_embedding = embedder.encode([question])
D, I = index.search(np.array(q_embedding), k=3)

# Get top 3 relevant chunks
retrieved_chunks = [texts[i] for i in I[0]]
context = "\n\n".join(retrieved_chunks)

# Load local model for answering
qa_model = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

# Generate answer
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
answer = qa_model(prompt, max_length=150, truncation=True)[0]["generated_text"]

print("\n📌 Answer:")
print(answer)