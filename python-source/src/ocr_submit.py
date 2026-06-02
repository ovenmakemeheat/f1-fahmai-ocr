from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .ocr_manifest import DEFAULT_SAMPLE_PATH


def load_parsed_predictions(parsed_dir: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    paths = sorted(parsed_dir.glob("*.json"))
    for path in tqdm(paths, desc="load parsed", unit="file"):
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"Parsed prediction must be a JSON object: {path}")
        predictions[path.stem] = value
    return predictions


def write_submission(
    parsed_dir: Path,
    output_path: Path,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    allow_missing: bool = False,
) -> None:
    predictions = load_parsed_predictions(parsed_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["artifact_id", "pred_json"])
        writer.writeheader()
        for row in tqdm(rows, desc="write submission", unit="row"):
            artifact_id = row["artifact_id"]
            prediction = predictions.get(artifact_id)
            if prediction is None:
                if not allow_missing:
                    raise FileNotFoundError(f"Missing parsed prediction for {artifact_id}")
                pred_json = "{}"
            else:
                pred_json = json.dumps(prediction, ensure_ascii=False, separators=(",", ":"))
            writer.writerow({"artifact_id": artifact_id, "pred_json": pred_json})
