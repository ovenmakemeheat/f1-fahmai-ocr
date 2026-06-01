# FahMai OCR End-to-End Pipeline

ระบบ Pipeline สำหรับรับและประมวลผล OCR งานแบบ Asynchronous เต็มรูปแบบ โครงสร้างถูกออกแบบมาเพื่อ:
1. **แยกการทำงาน (Decoupled)**: API รับ Request แล้วตอบกลับทันที ไม่ต้องรอดึงผลลัพธ์จากโมเดล
2. **รองรับโหลดสูง**: โมเดล vLLM (Typhoon OCR) รันแยกบน GPU (เช่น เครื่อง LANTA) ในขณะที่ API Server รันอยู่ที่ไหนก็ได้
3. **จัดเก็บผลลัพธ์เป็นระบบ**: เก็บข้อมูลทุกอย่างเข้า PostgreSQL พร้อมสถานะ `pending`, `processing`, `done`, `failed`

## โครงสร้างระบบ
```
ocr_pipeline/
├── docker-compose.yml       # รัน PostgreSQL และ FastAPI
├── .env.example             # ไฟล์ตั้งค่าตัวแปร (Config)
├── requirements.txt         # Dependencies สำหรับ FastAPI
├── Dockerfile               # Docker configuration สำหรับ FastAPI
├── db/
│   └── init.sql             # Schema Database และ Trigger (รันอัตโนมัติ)
└── app/
    ├── main.py              # FastAPI Router & Endpoints
    ├── config.py            # ดึงค่า Config จาก Environment Variables
    ├── database.py          # จัดการ Connection และคำสั่ง SQL
    ├── schemas.py           # Pydantic Models (ตรวจสอบ Data Type)
    ├── worker.py            # Background Task สำหรับเรียก vLLM
    └── services/
        ├── dataset.py       # อ่าน Metadata (JSON) ของ FahMai
        └── ocr.py           # ฟังก์ชันส่ง Prompt และรูปภาพเข้า vLLM API
```

## วิธีการติดตั้งและการใช้งาน

### 1. ตั้งค่า Environment Variables
เข้าไปที่โฟลเดอร์ `ocr_pipeline` แล้วสร้างไฟล์ `.env`
```bash
cp .env.example .env
```
เปิดไฟล์ `.env` และตั้งค่าต่างๆ ให้ถูกต้อง โดยเฉพาะ:
* `VLLM_BASE_URL`: ชี้ไปยัง Endpoint ของ vLLM (เช่น `http://192.168.1.100:8000/v1`)
* `PER_ARTIFACT_DIR`: ชี้ไปยังโฟลเดอร์ `/per_artifact` ของ Dataset (ใช้สำหรับดึงโครงสร้าง `visible_fields`)
* `POSTGRES_PASSWORD`: เปลี่ยนรหัสผ่าน Database

### 2. รัน vLLM Server (บนเครื่อง GPU)
ตัวอย่างคำสั่งเปิด vLLM:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model scb10x/typhoon-ocr-7b \
    --host 0.0.0.0 --port 8000 \
    --limit-mm-per-prompt image=1 \
    --max-model-len 8192
```

### 3. รัน API Server และ Database (ด้วย Docker Compose)
```bash
docker-compose up -d --build
```

---

## API Endpoints

### 1. `POST /ocr` (ส่งงาน)
ส่งรูปภาพ (Base64) และ `artifact_id` เข้าระบบ

**Request:**
```bash
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "VI-V-013-INV-2567-226313",
    "image_b64": "iVBORw0KGgoAAAANSUhEUgAA..."
  }'
```

**Response:** (ตอบกลับทันที ไม่บล็อกรอ OCR)
```json
{
  "artifact_id": "VI-V-013-INV-2567-226313",
  "status": "pending",
  "message": "Job accepted and queued for processing"
}
```

### 2. `GET /ocr/{artifact_id}` (เช็คสถานะและผลลัพธ์)
**Response:**
```json
{
  "artifact_id": "VI-V-013-INV-2567-226313",
  "doc_type": "vendor_invoice",
  "status": "done",
  "pred_json": {
    "vendor_id": "V-013",
    "payment_id": "INV-2567-226313",
    "total_amount_thb": "15000.00"
  },
  "created_at": "2024-11-20T10:00:00Z",
  "updated_at": "2024-11-20T10:00:15Z"
}
```

### 3. `GET /stats` (ดูสถิติระบบ)
**Response:**
```json
{
  "total": 100,
  "done": 85,
  "pending": 10,
  "processing": 3,
  "failed": 2
}
```

### 4. `GET /health` (ตรวจเช็คระบบ)
**Response:**
```json
{
  "status": "ok",
  "vllm_ok": true,
  "db_ok": true,
  "version": "1.0.0"
}
```
