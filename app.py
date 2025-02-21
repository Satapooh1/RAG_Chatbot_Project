from flask import Flask, render_template, request, session
from flask_session import Session
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv


app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("API Key สำหรับ Together.ai ยังไม่ได้ตั้งค่า! กรุณาเพิ่มใน .env")

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key=TOGETHER_API_KEY
)

embedding = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

loader_solar = TextLoader("solar_data.txt", encoding="utf-8")
docs_solar = loader_solar.load()
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=30)
docs_solar_spt = text_splitter.split_documents(docs_solar)
db_solar = FAISS.from_documents(docs_solar_spt, embedding)

loader_sea = TextLoader("sea_data.txt", encoding="utf-8")
docs_sea = loader_sea.load()
docs_sea_spt = text_splitter.split_documents(docs_sea)
db_sea = FAISS.from_documents(docs_sea_spt, embedding)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    คุณคือผู้ช่วยที่ตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้เท่านั้น
    หากไม่สามารถหาคำตอบจากข้อมูลได้ ให้ตอบว่า "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้"

    ข้อมูลอ้างอิง:
    {context}

    คำถาม:
    {question}
    คำตอบ:
    """
)

qa_chain_solar = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_solar.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt}
)

qa_chain_sea = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_sea.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt}
)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/solar_chat', methods=['GET', 'POST'])
def solar_chat():
    if 'chat_history_solar' not in session:
        session['chat_history_solar'] = []

    if request.method == 'POST':
        if 'reset' in request.form:
            session['chat_history_solar'] = []
        else:
            query = request.form['query']
            session['chat_history_solar'].append({'text': query, 'user': True})

            result = qa_chain_solar({'query': query})
            response = result.get('result', "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้")

            session['chat_history_solar'].append({'text': response, 'user': False})

    return render_template('chat.html', messages=session['chat_history_solar'], chat_title="Solar System Chatbot")

@app.route('/sea_chat', methods=['GET', 'POST'])
def sea_chat():
    if 'chat_history_sea' not in session:
        session['chat_history_sea'] = []

    if request.method == 'POST':
        if 'reset' in request.form:
            session['chat_history_sea'] = []
        else:
            query = request.form['query']
            session['chat_history_sea'].append({'text': query, 'user': True})

            result = qa_chain_sea({'query': query})
            response = result.get('result', "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้")

            session['chat_history_sea'].append({'text': response, 'user': False})

    return render_template('chat.html', messages=session['chat_history_sea'], chat_title="Sea Chatbot")

if __name__ == '__main__':
    app.run(debug=True)
