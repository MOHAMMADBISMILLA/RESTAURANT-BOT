from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os
from src.helper import load_pdf_file, load_docx_file, load_text_file, filter_to_minimal_docs, text_split
import tempfile
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from src.helper import rerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

embeddings = download_hugging_face_embeddings()

index_name = "restaurant-chatbot2"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2, "k": 4}
)

# Define prompt
prompt = PromptTemplate.from_template("""
You are a helpful restaurant assistant bot.
Use the following context to answer the user's question.
If the answer is not found in the context, reply: "Not found. No relevant information available."

Context:
{context}

Question:
{question}
""")

# Groq LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# === RAG Chain ===
rag_chain = RunnableParallel({
    "raw_docs": retriever,
    "question": RunnablePassthrough()
}) | (lambda inputs: {
    "question": inputs["question"],
    "context": "\n\n".join([doc.page_content for doc in rerank(inputs["question"], inputs["raw_docs"], top_k=3)])
}) | prompt | llm | StrOutputParser()

# ===================== Upload Document ======================
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

# ===================== Routes ======================
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("Input: ", input)

    try:
        response = rag_chain.invoke(input)
        return jsonify({"answer": str(response).strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
