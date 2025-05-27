from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

# Load a small local model (T5-base for Q&A)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# 1. Load a PDF
loader = PyPDFLoader(r"C:\Users\Sentry\Desktop\MyRAG\01.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Embedding using sentence-transformers
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)

# 4. Retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# 5. Ask questions
while True:
    query = input("\nAsk a question about the PDF (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print("\nAnswer:\n", result)