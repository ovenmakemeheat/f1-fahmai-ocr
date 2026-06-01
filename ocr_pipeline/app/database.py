# app/database.py
import asyncpg
from typing import Any
import json
from .config import get_settings

async def get_db_pool():
    settings = get_settings()
    pool = await asyncpg.create_pool(
        dsn=settings.asyncpg_dsn,
        min_size=1,
        max_size=10
    )
    return pool

async def init_db(pool):
    # สำหรับการเริ่มต้นอะไรเพิ่มเติมถ้าจำเป็น
    pass

async def create_job(pool: asyncpg.Pool, artifact_id: str, doc_type: str, image_b64: str) -> dict:
    query = """
        INSERT INTO ocr_jobs (artifact_id, doc_type, image_b64, status)
        VALUES ($1, $2, $3, 'pending')
        ON CONFLICT (artifact_id) DO UPDATE 
        SET status = 'pending', image_b64 = $3, updated_at = NOW()
        RETURNING id, artifact_id, doc_type, status
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, artifact_id, doc_type, image_b64)
        return dict(row)

async def update_job_processing(pool: asyncpg.Pool, artifact_id: str):
    query = "UPDATE ocr_jobs SET status = 'processing', updated_at = NOW() WHERE artifact_id = $1"
    async with pool.acquire() as conn:
        await conn.execute(query, artifact_id)

async def update_job_success(pool: asyncpg.Pool, artifact_id: str, pred_json: dict):
    query = """
        UPDATE ocr_jobs 
        SET status = 'done', pred_json = $2, updated_at = NOW() 
        WHERE artifact_id = $1
    """
    async with pool.acquire() as conn:
        await conn.execute(query, artifact_id, json.dumps(pred_json, ensure_ascii=False))

async def update_job_failed(pool: asyncpg.Pool, artifact_id: str, error_msg: str):
    query = """
        UPDATE ocr_jobs 
        SET status = 'failed', error_msg = $2, updated_at = NOW() 
        WHERE artifact_id = $1
    """
    async with pool.acquire() as conn:
        await conn.execute(query, artifact_id, error_msg)

async def get_job_status(pool: asyncpg.Pool, artifact_id: str) -> dict | None:
    query = "SELECT artifact_id, doc_type, status, pred_json, error_msg, created_at, updated_at FROM ocr_jobs WHERE artifact_id = $1"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, artifact_id)
        if row:
            # parse json
            result = dict(row)
            if isinstance(result['pred_json'], str):
                try:
                    result['pred_json'] = json.loads(result['pred_json'])
                except:
                    pass
            return result
        return None

async def get_stats(pool: asyncpg.Pool) -> dict:
    query = """
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'done') as done,
            COUNT(*) FILTER (WHERE status = 'pending') as pending,
            COUNT(*) FILTER (WHERE status = 'processing') as processing,
            COUNT(*) FILTER (WHERE status = 'failed') as failed
        FROM ocr_jobs
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query)
        return dict(row) if row else {"total":0, "done":0, "pending":0, "processing":0, "failed":0}
