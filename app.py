from flask import Flask, render_template, request
from langchain_community.embeddings import HuggingFaceEmbeddings  # ใช้สร้างเวกเตอร์จากข้อมูล
from langchain.vectorstores import FAISS  # ใช้ทำ vector database
from langchain.document_loaders import TextLoader  # โหลดไฟล์ข้อมูล
from langchain.text_splitter import CharacterTextSplitter  # ใช้แบ่งข้อความเป็นส่วนย่อย
from langchain.chains import RetrievalQA  # ใช้ทำ RAG
from langchain_together import ChatTogether  # ใช้โมเดลจาก Together.ai ผ่าน LangChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

loader_solar = TextLoader("solar_data.txt", encoding="utf-8")
docs_solar = loader_solar.load()

loader_sea = TextLoader("sea_data.txt", encoding="utf-8")
docs_sea = loader_sea.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_solar_spt = text_splitter.split_documents(docs_solar)
doc_sea_spt = text_splitter.split_documents(docs_sea)

db_solar = FAISS.from_documents(docs_solar_spt, embedding)
db_sea = FAISS.from_documents(doc_sea_spt, embedding)

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

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
    retriever=db_solar.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={"prompt": custom_prompt}
)

qa_chain_sea = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_sea.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={"prompt": custom_prompt}
)

chat_history_solar = []
chat_history_sea = []

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solar_chat', methods=['GET', 'POST'])
def solar_chat():
    global chat_history_solar
    if request.method == 'POST':
        if 'reset' in request.form:
            chat_history_solar = []
        else:
            query = request.form['query']
            chat_history_solar.append({'text': query, 'user': True})

            related_docs = qa_chain_solar.retriever.get_relevant_documents(query)
            if not related_docs:
                response = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้"
            else:
                result = qa_chain_solar({'query': query})
                response = result['result']

            chat_history_solar.append({'text': response, 'user': False})

    return render_template('chat.html', messages=chat_history_solar, chat_title="Solar System Chatbot")

@app.route('/sea_chat', methods=['GET', 'POST'])
def sea_chat():
    global chat_history_sea
    if request.method == 'POST':
        if 'reset' in request.form:
            chat_history_sea = []
        else:
            query = request.form['query']
            chat_history_sea.append({'text': query, 'user': True})

            related_docs = qa_chain_sea.retriever.get_relevant_documents(query)
            if not related_docs:
                response = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้"
            else:
                result = qa_chain_sea({'query': query})
                response = result['result']

            chat_history_sea.append({'text': response, 'user': False})

    return render_template('chat.html', messages=chat_history_sea, chat_title="Sea Chatbot")


if __name__ == '__main__':
    app.run(debug=True)
