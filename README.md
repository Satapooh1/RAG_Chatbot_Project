# แชทบอทด้วย RAG (Retrieval-Augmented Generation) โดยใช้ Flask และ LangChain

โปรเจกต์นี้เป็นแอปพลิเคชันแชทบอทที่ใช้ Retrieval-Augmented Generation (RAG) เพื่อดึงข้อมูลจากไฟล์ข้อความ และให้โมเดล LLaMA ตอบคำถามโดยอ้างอิงจากข้อมูลนั้น โดยมีแชทบอท 2 ตัว ได้แก่:

- **Solar System Chatbot**: ตอบคำถามเกี่ยวกับระบบสุริยะ โดยอ้างอิงข้อมูลจากไฟล์ `solar_data.txt`
- **Sea Chatbot**: ตอบคำถามเกี่ยวกับทะเลและสิ่งแวดล้อมทางทะเล โดยอ้างอิงข้อมูลจากไฟล์ `sea_data.txt`

## เทคโนโลยีที่ใช้
- **Python**
- **Flask** – สำหรับสร้างเว็บแอปพลิเคชัน
- **LangChain** – สำหรับทำ RAG และเชื่อมต่อกับโมเดล LLM
- **FAISS** – สำหรับสร้างเวกเตอร์ฐานข้อมูล
- **Together.ai** – สำหรับเรียกใช้งานโมเดล LLaMA

## โครงสร้างไฟล์
```
project_folder/
│
├── app.py                  # ไฟล์หลักสำหรับรัน Flask
├── solar_data.txt          # ข้อมูลเกี่ยวกับระบบสุริยะ
├── sea_data.txt            # ข้อมูลเกี่ยวกับทะเล
│
├── templates/              # ไฟล์ HTML
│   ├── index.html          # หน้าเลือกแชทบอท
│   └── chat.html           # หน้าแชท
│
└── static/                 # ไฟล์ CSS
    └── styles.css
```

## การติดตั้งไลบรารีที่จำเป็น
```bash
pip install flask flask_session langchain langchain_community langchain_together python-dotenv
```

## การตั้งค่า API Key
สร้างไฟล์ `.env` ในโฟลเดอร์เดียวกับ `app.py` และใส่
```
TOGETHER_API_KEY=YOUR_API_KEY
```

สมัครและรับ API Key ได้ที่ [https://www.together.ai](https://www.together.ai)

## การใช้งาน
1. รันแอปพลิเคชัน
```bash
python app.py
```

2. เปิดเบราว์เซอร์ไปที่
```
http://127.0.0.1:5000/
```

## คำอธิบายการทำงาน
- โหลดข้อมูลจากไฟล์ `solar_data.txt` และ `sea_data.txt`
- ใช้ `HuggingFaceEmbeddings` เพื่อแปลงข้อความเป็นเวกเตอร์
- ใช้ `FAISS` เพื่อสร้างเวกเตอร์ฐานข้อมูล
- ใช้ `RetrievalQA` เพื่อให้ LLaMA สร้างคำตอบจากข้อมูลที่ค้นคืนได้
- มีปุ่ม **"รีเซ็ต"** เพื่อเริ่มต้นแชทใหม่ (ต้องพิมพ์อะไรก็ได้ก่อนถึงจะกดลบได้)
- มีปุ่ม **"กลับหน้าแรก"** เพื่อกลับไปเลือกแชทบอท

## ข้อแนะนำเพิ่มเติม
- สามารถเพิ่มแชทบอทอื่น ๆ ได้โดยการเพิ่มไฟล์ข้อมูลและสร้าง route ใหม่
- ปรับแต่ง `chunk_size` และ `k` ให้เหมาะสมกับขนาดข้อมูล

