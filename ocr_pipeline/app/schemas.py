# app/schemas.py
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


# ─── Request ─────────────────────────────────────────────────

class OCRRequest(BaseModel):
    artifact_id: str = Field(..., description="ID ของเอกสาร เช่น VI-V-013-INV-2567-226313")
    image_b64: str   = Field(..., description="รูปภาพในรูปแบบ base64 string (ไม่ต้องมี data:image prefix)")
    doc_type: str | None = Field(None, description="ระบุ doc type ถ้าทราบ (ถ้าไม่ใส่จะ auto-detect จาก artifact_id)")


# ─── Response ────────────────────────────────────────────────

class OCRResponse(BaseModel):
    artifact_id: str
    doc_type: str
    status: str
    pred_json: dict[str, Any] | None = None
    error_msg: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class OCRSubmitResponse(BaseModel):
    artifact_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    vllm_ok: bool
    db_ok: bool
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    total: int
    done: int
    pending: int
    processing: int
    failed: int
