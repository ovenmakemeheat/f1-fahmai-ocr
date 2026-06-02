# app/worker.py
import asyncio
import logging
from .services.dataset import get_document_metadata
from .services.ocr import process_single_page, process_bank_header, process_bank_transactions

logger = logging.getLogger(__name__)

async def process_ocr_job(artifact_id: str, doc_type: str, image_b64: str) -> dict:
    try:
        category, pages = get_document_metadata(artifact_id, doc_type)
        if not category:
            category = "unknown"
            
        pred_json = {}
        
        # ── Bank statement (multi-page) ────────────────────────
        if category == "bank_statement":
            page_counter = 0
            for page in pages:
                kind = page.get("page_kind", "page")
                visible = page.get("visible_fields", [])
                source_rows = page.get("source_row_ids", [])
                
                if kind == "header":
                    result = await process_bank_header(image_b64, visible)
                    pred_json.update(result)
                else:
                    page_counter += 1
                    result = await process_bank_transactions(image_b64, page_counter, source_rows, visible)
                    pred_json.update(result)
                    
        # ── Single-page documents ─────────────────────────────
        else:
            if not pages:
                # Fallback if metadata not found
                raise ValueError(f"No metadata found for artifact {artifact_id} in {category}")
                
            page = pages[0]
            visible = page.get("visible_fields", [])
            result = await process_single_page(image_b64, visible, category)
            pred_json.update(result)
            
        logger.info(f"Successfully processed {artifact_id}")
        return pred_json
        
    except Exception as e:
        logger.error(f"Failed processing {artifact_id}: {str(e)}")
        raise e
