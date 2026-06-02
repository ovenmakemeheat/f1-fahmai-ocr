from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model/wangchanberta/label/20260602-171753/model"
DEFAULT_MAX_LENGTH = 256

app = FastAPI(
    title="Guardrail Pipeline API",
    description="Serves the finetuned WangchanBERT guardrail classifier.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int | None = Field(default=None, ge=8, le=512)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    max_length: int | None = Field(default=None, ge=8, le=512)


class Score(BaseModel):
    label: str
    score: float


class Prediction(BaseModel):
    text: str
    label: str
    label_id: int
    score: float
    is_attack: bool
    scores: list[Score]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    device: str
    loaded: bool


def resolve_model_path() -> Path:
    configured_path = os.getenv("GUARDRAIL_MODEL_PATH")
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return DEFAULT_MODEL_PATH


def resolve_device() -> torch.device:
    configured_device = os.getenv("GUARDRAIL_DEVICE", "auto").lower()
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured_device not in {"cpu", "cuda"}:
        raise ValueError("GUARDRAIL_DEVICE must be one of: auto, cpu, cuda")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("GUARDRAIL_DEVICE=cuda was requested, but CUDA is not available")
    return torch.device(configured_device)


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return {
        "model_path": model_path,
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "id2label": id2label,
    }


def get_max_length(requested_max_length: int | None) -> int:
    if requested_max_length is not None:
        return requested_max_length
    configured_max_length = os.getenv("GUARDRAIL_MAX_LENGTH")
    if configured_max_length:
        return int(configured_max_length)
    return DEFAULT_MAX_LENGTH


def predict_texts(texts: list[str], max_length: int | None = None) -> list[Prediction]:
    if not texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    cleaned_texts = [text.strip() for text in texts]
    if any(not text for text in cleaned_texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    try:
        resources = load_model()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model = resources["model"]
    tokenizer = resources["tokenizer"]
    device = resources["device"]
    id2label = resources["id2label"]

    encoded = tokenizer(
        cleaned_texts,
        truncation=True,
        max_length=get_max_length(max_length),
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu()

    predictions: list[Prediction] = []
    for text, row in zip(cleaned_texts, probabilities, strict=True):
        label_id = int(torch.argmax(row).item())
        label = str(id2label.get(label_id, label_id))
        score = float(row[label_id].item())
        scores = [
            Score(label=str(id2label.get(idx, idx)), score=float(value.item()))
            for idx, value in enumerate(row)
        ]
        predictions.append(
            Prediction(
                text=text,
                label=label,
                label_id=label_id,
                score=score,
                is_attack=label == "1" or label.lower() in {"attack", "prompt_injection"},
                scores=scores,
            )
        )
    return predictions


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_path = resolve_model_path()
    loaded = load_model.cache_info().currsize > 0
    device = os.getenv("GUARDRAIL_DEVICE", "auto")
    if loaded:
        device = str(load_model()["device"])
    return HealthResponse(
        status="ok",
        model_path=str(model_path),
        device=device,
        loaded=loaded,
    )


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    return predict_texts([request.text], max_length=request.max_length)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchPredictRequest) -> list[Prediction]:
    return predict_texts(request.texts, max_length=request.max_length)
