# FahMai OCR End-to-End Pipeline

ระบบ Pipeline สำหรับรับและประมวลผล OCR โครงสร้างถูกออกแบบมาเพื่อ:
1. **ประมวลผลทันที (Synchronous)**: API รับ Request แล้วเรียกใช้งานโมเดล ตอบกลับด้วยผลลัพธ์ใน HTTP Request เดียว
2. **รองรับสถาปัตยกรรมแยกส่วน**: โมเดล vLLM (Typhoon OCR) รันแยกบน GPU (เช่น เครื่อง LANTA) ในขณะที่ API Server รันอยู่ที่ไหนก็ได้

## โครงสร้างระบบ
```
ocr_pipeline/
├── docker-compose.yml       # รัน FastAPI
├── .env.example             # ไฟล์ตั้งค่าตัวแปร (Config)
├── requirements.txt         # Dependencies สำหรับ FastAPI
├── Dockerfile               # Docker configuration สำหรับ FastAPI
└── app/
    ├── main.py              # FastAPI Router & Endpoints
    ├── config.py            # ดึงค่า Config จาก Environment Variables
    ├── schemas.py           # Pydantic Models (ตรวจสอบ Data Type)
    ├── worker.py            # ฟังก์ชันหลักสำหรับเรียกและจัดการผลลัพธ์ vLLM
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

### 2. รัน vLLM Server (บนเครื่อง GPU)
ตัวอย่างคำสั่งเปิด vLLM:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model scb10x/typhoon-ocr-7b \
    --host 0.0.0.0 --port 8000 \
    --limit-mm-per-prompt image=1 \
    --max-model-len 8192
```

### 3. รัน API Server (ด้วย Docker Compose)
```bash
docker-compose up -d --build
```

---

## API Endpoints

### 1. `POST /ocr` (ส่งงานและรอรับผลลัพธ์)
ส่งรูปภาพ (Base64) และ `artifact_id` เข้าระบบและรอผลลัพธ์กลับมา

**Request:**
```bash
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "VI-V-013-INV-2567-226313",
    "image_b64": "iVBORw0KGgoAAAANSUhEUgAA..."
  }'
```

**Response:** (ประมวลผลและตอบกลับทันที)
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
  "error_msg": null
}
```

### 2. `GET /health` (ตรวจเช็คระบบ)
**Response:**
```json
{
  "status": "ok",
  "vllm_ok": true,
  "version": "1.0.0"
}
```
