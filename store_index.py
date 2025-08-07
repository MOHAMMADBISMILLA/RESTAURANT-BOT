import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


from src.helper import (
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
    load_pdf_file,
    load_docx_file,
    load_text_file
)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name ="restaurant-chatbot"

# Create Pinecone index (run only first time)
if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


documents = []
data_path = "data"

for filename in os.listdir(data_path):
    path = os.path.join(data_path, filename)
    if filename.endswith(".pdf"):
        docs = load_pdf_file(path)
    elif filename.endswith(".docx"):
        docs = load_docx_file(path)
    elif filename.endswith(".txt"):
        docs = load_text_file(path)
    else:
        print(f"Skipped unsupported file: {filename}")
        continue

    filtered = filter_to_minimal_docs(docs)
    documents.extend(filtered)

# Split and embed
text_chunks = text_split(documents)
embeddings = download_hugging_face_embeddings()
# Create vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
