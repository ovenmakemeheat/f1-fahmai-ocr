# app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from contextlib import asynccontextmanager
import logging
import httpx

from .config import get_settings
from .database import get_db_pool, create_job, get_job_status, get_stats
from .schemas import OCRRequest, OCRSubmitResponse, OCRResponse, HealthResponse, StatsResponse
from .worker import process_ocr_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store db pool globally
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await get_db_pool()
    logger.info("Database connection pool created")
    yield
    await db_pool.close()
    logger.info("Database connection pool closed")

app = FastAPI(title="FahMai OCR Pipeline API", lifespan=lifespan)

def get_pool():
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    return db_pool

@app.post("/ocr", response_model=OCRSubmitResponse)
async def submit_ocr_job(request: OCRRequest, background_tasks: BackgroundTasks, pool = Depends(get_pool)):
    """
    รับ POST request พร้อม JSON {artifact_id, image_b64, doc_type}
    บันทึกเข้า DB สถานะ pending และสั่ง process ใน background
    """
    doc_type = request.doc_type or "unknown"
    
    try:
        job = await create_job(pool, request.artifact_id, doc_type, request.image_b64)
        
        # ส่งเข้า background process
        background_tasks.add_task(
            process_ocr_job,
            pool,
            request.artifact_id,
            doc_type,
            request.image_b64
        )
        
        return OCRSubmitResponse(
            artifact_id=request.artifact_id,
            status="pending",
            message="Job accepted and queued for processing"
        )
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/ocr/{artifact_id}", response_model=OCRResponse)
async def get_ocr_result(artifact_id: str, pool = Depends(get_pool)):
    """
    เช็คสถานะและผลลัพธ์ของ OCR job ด้วย artifact_id
    """
    job = await get_job_status(pool, artifact_id)
    if not job:
        raise HTTPException(status_code=404, detail="Artifact not found")
        
    return OCRResponse(**job)

@app.get("/stats", response_model=StatsResponse)
async def get_system_stats(pool = Depends(get_pool)):
    """ดูสถิติของ OCR pipeline ทั้งหมด"""
    stats = await get_stats(pool)
    return StatsResponse(**stats)

@app.get("/health", response_model=HealthResponse)
async def health_check(pool = Depends(get_pool)):
    settings = get_settings()
    vllm_ok = False
    db_ok = False
    
    # Check DB
    try:
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
            db_ok = True
    except:
        pass
        
    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.vllm_base_url}/models")
            if r.status_code == 200:
                vllm_ok = True
    except:
        pass
        
    status = "ok" if (vllm_ok and db_ok) else "degraded"
    
    return HealthResponse(
        status=status,
        vllm_ok=vllm_ok,
        db_ok=db_ok
    )
