# app/services/dataset.py
import json
from pathlib import Path
from ..config import get_settings

def get_category_and_json(artifact_id: str) -> tuple[str | None, dict | None]:
    settings = get_settings()
    per_artifact_dir = Path(settings.per_artifact_dir)
    
    if not per_artifact_dir.exists():
        return None, None
        
    category_dirs = [d.name for d in per_artifact_dir.iterdir() if d.is_dir()]
    
    for cat in category_dirs:
        json_path = per_artifact_dir / cat / f"{artifact_id}.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                return cat, json.load(f)
                
    return None, None

def get_document_metadata(artifact_id: str, doc_type: str = None) -> tuple[str, list]:
    """
    Returns (category, list_of_pages)
    """
    category, metadata = get_category_and_json(artifact_id)
    if not category:
        category = doc_type or "unknown"
    
    pages = metadata.get("pages", []) if metadata else []
    return category, pages
