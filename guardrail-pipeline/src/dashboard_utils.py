from __future__ import annotations

from typing import Any

from src.guardrail_model import Prediction


def prediction_score(prediction: Prediction, label: str) -> float:
    for score in prediction.scores:
        if score.label == label:
            return score.score
    return 0.0


def prediction_row(
    prediction: Prediction,
    prediction_no: int,
    text: str,
    text_column: str = "text",
) -> dict[str, Any]:
    label_0_probability = prediction_score(prediction, "0")
    label_1_probability = prediction_score(prediction, "1")
    return {
        "prediction_no": prediction_no,
        text_column: text,
        "predicted_label": prediction.label,
        "predicted_label_id": prediction.label_id,
        "predicted_score": prediction.score,
        "label_0_probability": label_0_probability,
        "label_1_probability": label_1_probability,
        "label_confidence": max(label_0_probability, label_1_probability),
        "attack_score": prediction.attack_score,
        "threshold": prediction.threshold,
        "is_attack": prediction.is_attack,
        "message": prediction.message,
        "total_token": prediction.total_token,
    }


def prediction_rows(
    predictions: list[Prediction],
    texts: list[str],
    text_column: str,
) -> list[dict[str, Any]]:
    return [
        prediction_row(
            prediction,
            prediction_no=index + 1,
            text=text,
            text_column=text_column,
        )
        for index, (prediction, text) in enumerate(zip(predictions, texts, strict=True))
    ]
