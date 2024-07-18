from flask import Flask, request, render_template, jsonify, session
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

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

def process_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    processed_files = session.get('processed_files', [])
    return render_template('index.html', processed_files=processed_files)

@app.route('/process', methods=['POST'])
def process_pdfs():
    pdf_files = request.files.getlist('pdf_files')
    if pdf_files:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Store processed file names in session
        processed_files = session.get('processed_files', [])
        for pdf in pdf_files:
            processed_files.append(pdf.filename)
        session['processed_files'] = processed_files

        return jsonify({'message': 'Processing complete'})
    else:
        return jsonify({'message': 'No files uploaded'})

@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.form.get('user_question')
    if user_question:
        response = process_question(user_question)
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'No question asked'})

if __name__ == '__main__':
    app.run(debug=True)
