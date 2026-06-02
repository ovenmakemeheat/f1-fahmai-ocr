# app/main.py
from fastapi import FastAPI, HTTPException
import logging
import httpx

from .config import get_settings
from .schemas import OCRRequest, OCRResponse, HealthResponse
from .worker import process_ocr_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FahMai OCR Pipeline API")

@app.post("/ocr", response_model=OCRResponse)
async def submit_ocr_job(request: OCRRequest):
    """
    รับ POST request พร้อม JSON {artifact_id, image_b64, doc_type}
    ทำการประมวลผล OCR และคืนค่าผลลัพธ์แบบ Synchronous
    """
    doc_type = request.doc_type or "unknown"
    
    try:
        pred_json = await process_ocr_job(
            request.artifact_id,
            doc_type,
            request.image_b64
        )
        
        return OCRResponse(
            artifact_id=request.artifact_id,
            doc_type=doc_type,
            status="done",
            pred_json=pred_json
        )
    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        return OCRResponse(
            artifact_id=request.artifact_id,
            doc_type=doc_type,
            status="failed",
            error_msg=str(e)
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    vllm_ok = False
    
    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.vllm_base_url}/models")
            if r.status_code == 200:
                vllm_ok = True
    except:
        pass
        
    status = "ok" if vllm_ok else "degraded"
    
    return HealthResponse(
        status=status,
        vllm_ok=vllm_ok
    )
