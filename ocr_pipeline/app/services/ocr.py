# app/services/ocr.py
import httpx
import json
from ..config import get_settings
import logging

logger = logging.getLogger(__name__)

async def parse_json_response(raw: str, fallback_fields: list[str]) -> dict | list:
    """Parse JSON อย่างปลอดภัย"""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                raw = part
                break
    
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e+1])
            except Exception:
                pass
                
    # fallback
    return {f: "" for f in fallback_fields}

async def call_vllm(b64_image: str, prompt: str, max_tokens: int = 1024) -> str:
    settings = get_settings()
    
    # Ensure correct mime prefix
    if not b64_image.startswith("data:image"):
        b64_image = f"data:image/png;base64,{b64_image}"
        
    payload = {
        "model": settings.vllm_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_image}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    async with httpx.AsyncClient(timeout=settings.vllm_timeout) as client:
        try:
            response = await client.post(
                f"{settings.vllm_base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"vLLM API Error: {str(e)}")
            raise e

async def process_single_page(b64_image: str, visible_fields: list[str], doc_type: str) -> dict:
    fields_list = "\n".join(f"- {f}" for f in visible_fields)
    prompt = f"""You are an OCR assistant. Extract the following fields from this document image.
Document type: {doc_type}

Fields to extract:
{fields_list}

Reply ONLY with a valid JSON object. Use exactly these keys.
Use empty string "" if a field cannot be read.
No markdown, no explanation. JSON only."""

    settings = get_settings()
    raw_response = await call_vllm(b64_image, prompt, max_tokens=settings.vllm_max_tokens)
    
    result = await parse_json_response(raw_response, visible_fields)
    if not isinstance(result, dict):
        result = {f: "" for f in visible_fields}
        
    for f in visible_fields:
        if f not in result:
            result[f] = ""
            
    return result

async def process_bank_header(b64_image: str, visible_fields: list[str]) -> dict:
    raw = await process_single_page(b64_image, visible_fields, "bank statement header")
    return {f"L0_{k}": str(v) for k, v in raw.items()}

async def process_bank_transactions(b64_image: str, page_num: int, source_rows: list[str], visible_fields: list[str]) -> dict:
    fields_str = ", ".join(visible_fields)
    n_rows = len(source_rows)
    
    prompt = f"""You are an OCR assistant analyzing a bank statement transaction page.
Extract ALL transaction rows visible on this page.

For each row, extract these fields: {fields_str}
There are approximately {n_rows} rows. Return them in order top to bottom.

Reply ONLY with a valid JSON array of objects. No markdown. No explanation."""

    settings = get_settings()
    raw_response = await call_vllm(b64_image, prompt, max_tokens=settings.vllm_max_tokens_tx)
    
    rows = await parse_json_response(raw_response, [])
    if not isinstance(rows, list):
        rows = []
        
    result = {}
    for i, row_id in enumerate(source_rows):
        row_data = rows[i] if i < len(rows) else {}
        for field in visible_fields:
            key = f"L{page_num}_{row_id}_{field}"
            result[key] = str(row_data.get(field, ""))
            
    return result
