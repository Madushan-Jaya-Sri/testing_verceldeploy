import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure GoogleGenerativeAIEmbeddings with API key
from google.generativeai import configure
configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# Global variable to store processed text chunks
processed_text_chunks = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
    the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    context:\n{context}\n
    question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_and_store_pdf(pdf_docs):
    global processed_text_chunks
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    processed_text_chunks = text_chunks

@app.route('/process', methods=['POST'])
def process_pdf():
    global processed_text_chunks
    pdf_docs = request.files.getlist('pdf_files')
    process_and_store_pdf(pdf_docs)
    return jsonify({'message': 'Processing complete'})

@app.route('/ask', methods=['POST'])
def ask_question():
    global processed_text_chunks
    user_question = request.form['user_question']
    
    # Check if text chunks are processed
    if processed_text_chunks is None:
        return jsonify({'error': 'PDFs not processed yet'})

    # Use the stored text chunks to find relevant documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.from_texts(processed_text_chunks, embedding=embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return jsonify({'response': response["output_text"]})
    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"})

if __name__ == '__main__':
    app.run(debug=True)
