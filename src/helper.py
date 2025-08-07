from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from pathlib import Path
from typing import List, Union

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

# Load all supported files from directory
# def load_all_documents_from_directory(directory_path: Union[str, Path]) -> List[Document]:
#     all_docs: List[Document] = []
#     directory = Path(directory_path)

#     for file_path in directory.iterdir():
#         if file_path.suffix == ".pdf":
#             loader = PyPDFLoader(str(file_path))
#         elif file_path.suffix == ".docx":
#             loader = Docx2txtLoader(str(file_path))
#         elif file_path.suffix == ".txt":
#             loader = TextLoader(str(file_path))
#         else:
#             continue  # Skip unsupported file formats
#         docs = loader.load()
#         all_docs.extend(docs)

#     return all_docs

def load_pdf_file(data: str) -> List[Document]:
    loader = PyPDFLoader(data)
    return loader.load()

def load_docx_file(path: str) -> List[Document]:
    loader = Docx2txtLoader(path)
    return loader.load()

def load_text_file(path: str) -> List[Document]:
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs


def text_split(extracted_data: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def rerank(query, docs, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]
