from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "model"
DEFAULT_MODEL_ID = "microhum/wangchanberta-fahmai-guardrails-v1"
MODEL_VARIANTS = {
    DEFAULT_MODEL_NAME: DEFAULT_MODEL_ID,
    "wangchanberta-fahmai-v1": DEFAULT_MODEL_ID,
    DEFAULT_MODEL_ID: DEFAULT_MODEL_ID,
}
DEFAULT_MAX_LENGTH = 510
DEFAULT_ATTACK_THRESHOLD = 0.75
DEFAULT_DASHBOARD_BATCH_SIZE = 32
INJECTION_MESSAGE = "ไม่สามารถระบุได้"
MIN_MAX_LENGTH = 8
UNBOUNDED_TOKENIZER_LENGTH = 100_000


class Score(BaseModel):
    label: str
    score: float


class Prediction(BaseModel):
    text: str
    label: str
    label_id: int
    score: float
    attack_score: float
    threshold: float
    is_attack: bool
    message: str
    total_token: int
    scores: list[Score]


def get_model_variants() -> dict[str, str]:
    variants = dict(MODEL_VARIANTS)

    configured_model_id = os.getenv("GUARDRAIL_MODEL_ID")
    if configured_model_id:
        variants[DEFAULT_MODEL_NAME] = configured_model_id

    configured_path = os.getenv("GUARDRAIL_MODEL_PATH")
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        variants["local"] = str(path)

    return variants


def resolve_model_id(model_name: str | None = None) -> str:
    requested_model = (model_name or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    variants = get_model_variants()

    if requested_model in variants:
        return variants[requested_model]

    allowed_models = ", ".join(sorted(variants))
    raise ValueError(
        f"Unknown model variant '{requested_model}'. Available variants: {allowed_models}"
    )


def resolve_device() -> torch.device:
    configured_device = os.getenv("GUARDRAIL_DEVICE", "auto").lower()
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured_device not in {"cpu", "cuda"}:
        raise ValueError("GUARDRAIL_DEVICE must be one of: auto, cpu, cuda")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("GUARDRAIL_DEVICE=cuda was requested, but CUDA is not available")
    return torch.device(configured_device)


@lru_cache(maxsize=4)
def load_model(model_id: str) -> dict[str, Any]:
    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return {
        "model_id": model_id,
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "id2label": id2label,
    }


def get_model_max_length(resources: dict[str, Any]) -> int:
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    configured_max_length = getattr(tokenizer, "model_max_length", DEFAULT_MAX_LENGTH)
    if configured_max_length > UNBOUNDED_TOKENIZER_LENGTH:
        configured_max_length = DEFAULT_MAX_LENGTH

    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if max_position_embeddings:
        padding_idx = getattr(model.config, "pad_token_id", 0) or 0
        configured_max_length = min(
            configured_max_length,
            max_position_embeddings - padding_idx,
        )

    return max(MIN_MAX_LENGTH, int(configured_max_length))


def get_max_length(requested_max_length: int | None, resources: dict[str, Any]) -> int:
    model_max_length = get_model_max_length(resources)
    if requested_max_length is not None:
        if requested_max_length < MIN_MAX_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"max_length must be at least {MIN_MAX_LENGTH}",
            )
        return min(requested_max_length, model_max_length)

    configured_max_length = os.getenv("GUARDRAIL_MAX_LENGTH")
    if configured_max_length:
        return min(int(configured_max_length), model_max_length)
    return min(DEFAULT_MAX_LENGTH, model_max_length)


def get_threshold(requested_threshold: float | None) -> float:
    if requested_threshold is not None:
        threshold = requested_threshold
    else:
        configured_threshold = os.getenv("GUARDRAIL_ATTACK_THRESHOLD")
        threshold = (
            float(configured_threshold)
            if configured_threshold
            else DEFAULT_ATTACK_THRESHOLD
        )

    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    return threshold


def get_attack_label_ids(id2label: dict[int, str]) -> set[int]:
    return {
        idx
        for idx, label in id2label.items()
        if str(label) == "1" or str(label).lower() in {"attack", "prompt_injection"}
    }


def load_resources_for_request(model_name: str | None) -> tuple[str, dict[str, Any]]:
    try:
        model_id = resolve_model_id(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return model_id, load_model(model_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def predict_texts(
    texts: list[str],
    model_name: str | None = None,
    max_length: int | None = None,
    threshold: float | None = None,
) -> list[Prediction]:
    if not texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    cleaned_texts = [text.strip() for text in texts]
    if any(not text for text in cleaned_texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    _, resources = load_resources_for_request(model_name)
    model = resources["model"]
    tokenizer = resources["tokenizer"]
    device = resources["device"]
    id2label = resources["id2label"]
    attack_threshold = get_threshold(threshold)
    attack_label_ids = get_attack_label_ids(id2label)
    effective_max_length = get_max_length(max_length, resources)

    encoded = tokenizer(
        cleaned_texts,
        truncation=True,
        max_length=effective_max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    try:
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu()
    except RuntimeError as exc:
        if device.type == "cuda":
            raise HTTPException(
                status_code=500,
                detail=(
                    "CUDA inference failed. Restart the API process before retrying; "
                    f"inputs are capped at max_length={effective_max_length}."
                ),
            ) from exc
        raise

    predictions: list[Prediction] = []
    input_lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
    for text, row, total_token in zip(cleaned_texts, probabilities, input_lengths, strict=True):
        label_id = int(torch.argmax(row).item())
        label = str(id2label.get(label_id, label_id))
        score = float(row[label_id].item())
        attack_score = (
            float(sum(row[idx].item() for idx in attack_label_ids))
            if attack_label_ids
            else float(row[1].item() if len(row) > 1 else 0.0)
        )
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
                attack_score=attack_score,
                threshold=attack_threshold,
                is_attack=attack_score >= attack_threshold,
                message=INJECTION_MESSAGE if label_id == 1 else "",
                total_token=int(total_token),
                scores=scores,
            )
        )
    return predictions


def predict_texts_batched(
    texts: list[str],
    model_name: str | None = None,
    max_length: int | None = None,
    threshold: float | None = None,
    batch_size: int = DEFAULT_DASHBOARD_BATCH_SIZE,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for start in range(0, len(texts), batch_size):
        predictions.extend(
            predict_texts(
                texts[start : start + batch_size],
                model_name=model_name,
                max_length=max_length,
                threshold=threshold,
            )
        )
    return predictions
