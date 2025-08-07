from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from src.helper import load_pdf_file, load_docx_file, load_text_file, filter_to_minimal_docs, text_split
import tempfile
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from src.helper import rerank
from langchain.chains import LLMChain
from src.prompt import prompt
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name ="restaurant-chatbot2"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4}
)




@app.route("/upload_document", methods=["POST"])
def upload_document():
    if 'pdf' not in request.files and 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files.get('file') or request.files.get('pdf')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = os.path.join(tmp_dir, filename)
        file.save(temp_path)

        try:
            if ext == ".pdf":
                docs = load_pdf_file(temp_path)
            elif ext == ".docx":
                docs = load_docx_file(temp_path)
            elif ext == ".txt":
                docs = load_text_file(temp_path)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            filtered_docs = filter_to_minimal_docs(docs)
            chunks = text_split(filtered_docs)
            docsearch.add_documents(chunks)

            return jsonify({"message": f"Document '{filename}' indexed successfully!"}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500




@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("Input: ", input)

    # Step 1: Get top documents from retriever
    raw_docs = retriever.get_relevant_documents(input)

    if not raw_docs:
        return jsonify({"answer": "Not found. No relevant information available."}), 200

    # Step 2: Rerank using BGE reranker
    top_docs = rerank(input, raw_docs, top_k=3)

    chatModel = ChatGroq(model="openai/gpt-oss-120b",temperature=0)

    chain = LLMChain(llm=chatModel, prompt=prompt, output_parser=StrOutputParser())
    response = chain.invoke({"context": top_docs, "question": input})

    return jsonify({"answer": str(response).strip()})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
