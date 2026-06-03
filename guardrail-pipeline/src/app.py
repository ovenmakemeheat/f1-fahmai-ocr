from __future__ import annotations

import html
import io
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model/wangchanberta/label/20260602-171753/model"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_ATTACK_THRESHOLD = 0.75
DEFAULT_DASHBOARD_BATCH_SIZE = 32

DASHBOARD_DOWNLOADS: dict[str, str] = {}

app = FastAPI(
    title="Guardrail Pipeline API",
    description="Serves the finetuned WangchanBERT guardrail classifier.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int | None = None
    threshold: float | None = None


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    max_length: int | None = None
    threshold: float | None = None


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
        if requested_max_length < 8:
            raise HTTPException(status_code=400, detail="max_length must be at least 8")
        return requested_max_length
    configured_max_length = os.getenv("GUARDRAIL_MAX_LENGTH")
    if configured_max_length:
        return int(configured_max_length)
    return DEFAULT_MAX_LENGTH


def get_threshold(requested_threshold: float | None) -> float:
    if requested_threshold is not None:
        threshold = requested_threshold
    else:
        configured_threshold = os.getenv("GUARDRAIL_ATTACK_THRESHOLD")
        threshold = float(configured_threshold) if configured_threshold else DEFAULT_ATTACK_THRESHOLD

    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    return threshold


def predict_texts(
    texts: list[str],
    max_length: int | None = None,
    threshold: float | None = None,
) -> list[Prediction]:
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
    attack_threshold = get_threshold(threshold)
    attack_label_ids = {
        idx
        for idx, label in id2label.items()
        if str(label) == "1" or str(label).lower() in {"attack", "prompt_injection"}
    }

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
                scores=scores,
            )
        )
    return predictions


def predict_texts_batched(
    texts: list[str],
    max_length: int | None = None,
    threshold: float | None = None,
    batch_size: int = DEFAULT_DASHBOARD_BATCH_SIZE,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for start in range(0, len(texts), batch_size):
        predictions.extend(
            predict_texts(
                texts[start : start + batch_size],
                max_length=max_length,
                threshold=threshold,
            )
        )
    return predictions


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def format_float(value: float) -> str:
    return f"{value:.3f}"


def make_distribution(items: pd.Series, limit: int = 12) -> list[dict[str, Any]]:
    counts = items.astype(str).fillna("").value_counts(dropna=False).head(limit)
    total = max(1, int(len(items)))
    return [
        {
            "label": str(label),
            "count": int(count),
            "percent": float(count / total),
        }
        for label, count in counts.items()
    ]


def make_numeric_bins(values: pd.Series, bins: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    total = max(1, int(len(numeric)))
    distribution = []
    for label, lower, upper in bins:
        count = int(((numeric >= lower) & (numeric < upper)).sum())
        distribution.append({"label": label, "count": count, "percent": float(count / total)})
    return distribution


def make_text_length_bins(lengths: pd.Series) -> list[dict[str, Any]]:
    return make_numeric_bins(
        lengths,
        [
            ("0-80", 0, 80),
            ("80-160", 80, 160),
            ("160-320", 160, 320),
            ("320-640", 320, 640),
            ("640+", 640, float("inf")),
        ],
    )


def make_threshold_curve(output: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
        count = int((output["attack_score"] >= threshold).sum())
        rows.append(
            {
                "label": f"{threshold:.2f}",
                "count": count,
                "percent": float(count / max(1, len(output))),
            }
        )
    return rows


def make_confusion_counts(output: pd.DataFrame) -> list[dict[str, Any]]:
    if "true_is_attack" not in output.columns:
        return []
    cases = [
        ("true normal / predicted normal", (~output["true_is_attack"]) & (~output["is_attack"])),
        ("true normal / predicted attack", (~output["true_is_attack"]) & output["is_attack"]),
        ("true attack / predicted normal", output["true_is_attack"] & (~output["is_attack"])),
        ("true attack / predicted attack", output["true_is_attack"] & output["is_attack"]),
    ]
    total = max(1, len(output))
    return [
        {"label": label, "count": int(mask.sum()), "percent": float(mask.sum() / total)}
        for label, mask in cases
    ]


def group_wrong_distribution(output: pd.DataFrame, column: str, limit: int = 12) -> list[dict[str, Any]]:
    if column not in output.columns or "is_wrong" not in output.columns:
        return []
    grouped = (
        output.groupby(column, dropna=False)["is_wrong"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .sort_values(["sum", "mean"], ascending=False)
        .head(limit)
    )
    return [
        {
            "label": str(row[column]),
            "count": int(row["sum"]),
            "percent": float(row["mean"]),
            "rows": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


def profile_columns(frame: pd.DataFrame, text_column: str, label_column: str | None) -> list[dict[str, Any]]:
    profiles = []
    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        missing = int(series.isna().sum())
        profile = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "missing": missing,
            "unique": unique,
            "role": "text" if column == text_column else "label" if column == label_column else "metadata",
        }
        if column == text_column:
            text_lengths = series.astype(str).str.len()
            profile["avg_length"] = format_float(float(text_lengths.mean()))
            profile["max_length"] = int(text_lengths.max())
        else:
            profile["top_values"] = ", ".join(
                f"{label}: {count}" for label, count in series.astype(str).value_counts().head(3).items()
            )
        profiles.append(profile)
    return profiles


def render_distribution(title: str, distribution: list[dict[str, Any]]) -> str:
    if not distribution:
        return ""
    bars = "".join(
        f"""
        <div class="bar-row">
          <span class="bar-label">{escape(item["label"])}</span>
          <div class="bar-track"><div class="bar-fill" style="width: {item["percent"] * 100:.1f}%"></div></div>
          <span class="bar-count">{escape(item["count"])}</span>
        </div>
        """
        for item in distribution
    )
    return f"""
    <div class="chart-card">
      <div class="chart-title">{escape(title)}</div>
      {bars}
    </div>
    """


def render_rate_distribution(title: str, distribution: list[dict[str, Any]]) -> str:
    if not distribution:
        return ""
    bars = "".join(
        f"""
        <div class="bar-row">
          <span class="bar-label">{escape(item["label"])}</span>
          <div class="bar-track"><div class="bar-fill danger-fill" style="width: {item["percent"] * 100:.1f}%"></div></div>
          <span class="bar-count">{item["count"]}/{item.get("rows", "")}</span>
        </div>
        """
        for item in distribution
    )
    return f"""
    <div class="chart-card">
      <div class="chart-title">{escape(title)}</div>
      {bars}
    </div>
    """


def render_compact_table(
    title: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    description: str = "",
) -> str:
    if not rows:
        return ""
    description_html = f'<p class="section-note">{escape(description)}</p>' if description else ""
    headers = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{render_cell(column, row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"""
    <section>
      <div class="section-title">{escape(title)}</div>
      {description_html}
      <div class="table-wrap">
        <table>
          <thead><tr>{headers}</tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </section>
    """


def render_cell(column: str, value: Any) -> str:
    if column == "prediction_no":
        return f'<span class="row-id">{escape(value)}</span>'
    if column == "is_attack":
        badge_class = "attack" if str(value).lower() == "true" else "normal"
        label = "attack" if badge_class == "attack" else "normal"
        return f'<span class="badge {badge_class}">{label}</span>'
    if column == "is_wrong":
        badge_class = "wrong" if str(value).lower() == "true" else "correct"
        label = "wrong" if badge_class == "wrong" else "correct"
        return f'<span class="badge {badge_class}">{label}</span>'
    if column == "attack_score":
        try:
            score = float(value)
        except (TypeError, ValueError):
            return escape(value)
        return (
            f'<div class="score-cell"><span>{score:.3f}</span>'
            f'<div class="mini-track"><div class="mini-fill" style="width: {score * 100:.1f}%"></div></div></div>'
        )
    if isinstance(value, float):
        return format_float(value)
    return escape(value)


def render_dashboard(error: str | None = None, result: dict[str, Any] | None = None) -> str:
    threshold = escape(os.getenv("GUARDRAIL_ATTACK_THRESHOLD", str(DEFAULT_ATTACK_THRESHOLD)))
    max_length = escape(os.getenv("GUARDRAIL_MAX_LENGTH", str(DEFAULT_MAX_LENGTH)))
    error_html = f'<div class="alert error">{escape(error)}</div>' if error else ""
    result_html = ""

    if result:
        metrics = result["metrics"]
        metric_cards = "".join(
            f'<div class="metric"><span>{escape(name)}</span><strong>{escape(value)}</strong></div>'
            for name, value in metrics.items()
        )
        table_rows = "".join(
            "<tr>"
            + "".join(f"<td>{render_cell(column, row.get(column, ''))}</td>" for column in result["columns"])
            + "</tr>"
            for row in result["preview_rows"]
        )
        headers = "".join(f"<th>{escape(column)}</th>" for column in result["columns"])
        download_html = (
            f'<a class="button secondary" href="/dashboard/download/{escape(result["download_id"])}">'
            "Download predictions CSV</a>"
        )
        wrong_html = ""
        if result["wrong_rows"]:
            wrong_rows = "".join(
                "<tr>"
                + "".join(
                    f"<td>{render_cell(column, row.get(column, ''))}</td>"
                    for column in result["wrong_columns"]
                )
                + "</tr>"
                for row in result["wrong_rows"]
            )
            wrong_headers = "".join(f"<th>{escape(column)}</th>" for column in result["wrong_columns"])
            wrong_html = f"""
            <section>
              <div class="section-title">Wrong Predictions</div>
              <div class="table-wrap">
                <table>
                  <thead><tr>{wrong_headers}</tr></thead>
                  <tbody>{wrong_rows}</tbody>
                </table>
              </div>
            </section>
            """

        profile_headers = ["role", "column", "dtype", "non_null", "missing", "unique", "avg_length", "max_length", "top_values"]
        profile_rows = "".join(
            "<tr>"
            + "".join(f"<td>{escape(row.get(column, ''))}</td>" for column in profile_headers)
            + "</tr>"
            for row in result["column_profiles"]
        )
        profile_header_html = "".join(f"<th>{escape(column)}</th>" for column in profile_headers)
        charts_html = "".join(
            [
                render_distribution("Predicted class", result["prediction_distribution"]),
                render_distribution("Attack decision", result["attack_distribution"]),
                render_distribution("True label", result["label_distribution"]),
                render_distribution("Column: category", result["category_distribution"]),
                render_distribution("Confidence bands", result["confidence_distribution"]),
                render_distribution("Text length", result["text_length_distribution"]),
                render_distribution("Threshold impact", result["threshold_curve"]),
                render_distribution("Confusion matrix counts", result["confusion_distribution"]),
                render_rate_distribution("Wrong by category", result["wrong_by_category"]),
                render_rate_distribution("Wrong by source_file", result["wrong_by_source_file"]),
            ]
        )
        risky_html = render_compact_table(
            "Highest Attack-Score Rows",
            result["risky_rows"],
            result["risky_columns"],
            "Rows sorted by attack_score from highest to lowest. These are the requests the model considers most likely to be attacks, even if the final decision changes when you adjust the threshold.",
        )
        low_confidence_html = render_compact_table(
            "Lowest Prediction-Confidence Rows",
            result["low_confidence_rows"],
            result["risky_columns"],
            "Rows sorted by predicted_score from lowest to highest. These are the cases where the model is least confident in its chosen class and are useful for manual review or dataset improvement.",
        )

        result_html = f"""
        <section>
          <div class="section-title">Results</div>
          <div class="metrics">{metric_cards}</div>
          <div class="actions">{download_html}</div>
          <div class="chart-grid">{charts_html}</div>
        </section>
        {risky_html}
        {low_confidence_html}
        <section>
          <div class="section-title">Column Profile</div>
          <div class="table-wrap">
            <table>
              <thead><tr>{profile_header_html}</tr></thead>
              <tbody>{profile_rows}</tbody>
            </table>
          </div>
        </section>
        <section>
          <div class="section-title">Prediction Rows</div>
          <div class="toolbar">
            <input id="tableFilter" type="text" placeholder="Filter visible rows..." oninput="filterRows()">
            <button type="button" class="secondary" onclick="showOnlyWrong()">Show wrong</button>
            <button type="button" class="secondary" onclick="showOnlyAttack()">Show attacks</button>
            <button type="button" class="secondary" onclick="showAllRows()">Show all</button>
          </div>
          <div class="table-wrap">
            <table id="predictionTable">
              <thead><tr>{headers}</tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </section>
        {wrong_html}
        """

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Guardrail Dashboard</title>
      <style>
        :root {{
          color-scheme: light;
          --bg: #f6f7f9;
          --panel: #ffffff;
          --text: #17202a;
          --muted: #64748b;
          --border: #d9e0e8;
          --accent: #0f766e;
          --accent-dark: #115e59;
          --danger: #b91c1c;
          --ok: #166534;
          --warn-bg: #fff7ed;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          background: var(--bg);
          color: var(--text);
          font-family: Arial, Helvetica, sans-serif;
        }}
        main {{
          max-width: 1180px;
          margin: 0 auto;
          padding: 28px 18px 48px;
        }}
        h1 {{
          margin: 0 0 6px;
          font-size: 28px;
          letter-spacing: 0;
        }}
        .subhead {{
          margin: 0 0 22px;
          color: var(--muted);
          font-size: 14px;
        }}
        section {{
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 18px;
          margin-top: 18px;
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 14px;
        }}
        label {{
          display: block;
          color: var(--muted);
          font-size: 12px;
          font-weight: 700;
          margin-bottom: 6px;
        }}
        input[type="text"], input[type="number"], input[type="file"] {{
          width: 100%;
          min-height: 38px;
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 8px 10px;
          background: #fff;
          color: var(--text);
        }}
        .checkbox {{
          display: flex;
          align-items: center;
          gap: 8px;
          min-height: 38px;
          color: var(--text);
          font-size: 14px;
        }}
        .checkbox input {{ width: 16px; height: 16px; }}
        .actions {{
          display: flex;
          gap: 10px;
          align-items: center;
          margin-top: 16px;
          flex-wrap: wrap;
        }}
        button, .button {{
          border: 0;
          border-radius: 6px;
          background: var(--accent);
          color: white;
          min-height: 38px;
          padding: 9px 14px;
          font-weight: 700;
          text-decoration: none;
          cursor: pointer;
        }}
        button:hover, .button:hover {{ background: var(--accent-dark); }}
        .secondary {{
          background: #334155;
        }}
        .secondary:hover {{ background: #1f2937; }}
        .hint {{
          color: var(--muted);
          font-size: 13px;
        }}
        .alert {{
          border-radius: 6px;
          padding: 12px;
          margin: 14px 0;
          font-size: 14px;
        }}
        .error {{
          color: var(--danger);
          background: #fee2e2;
          border: 1px solid #fecaca;
        }}
        .section-title {{
          font-size: 16px;
          font-weight: 800;
          margin-bottom: 12px;
        }}
        .section-note {{
          margin: -4px 0 12px;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.45;
        }}
        .metrics {{
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 10px;
        }}
        .metric {{
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 12px;
          background: #fbfdff;
        }}
        .metric span {{
          display: block;
          color: var(--muted);
          font-size: 12px;
          margin-bottom: 6px;
        }}
        .metric strong {{ font-size: 20px; }}
        .chart-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
          margin-top: 16px;
        }}
        .chart-card {{
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 12px;
          background: #fbfdff;
        }}
        .chart-title {{
          font-size: 13px;
          font-weight: 800;
          margin-bottom: 10px;
        }}
        .bar-row {{
          display: grid;
          grid-template-columns: minmax(80px, 160px) 1fr 48px;
          gap: 8px;
          align-items: center;
          margin: 7px 0;
        }}
        .bar-label, .bar-count {{
          color: #334155;
          font-size: 12px;
          overflow-wrap: anywhere;
        }}
        .bar-count {{ text-align: right; }}
        .bar-track, .mini-track {{
          height: 8px;
          background: #e2e8f0;
          border-radius: 999px;
          overflow: hidden;
        }}
        .bar-fill, .mini-fill {{
          height: 100%;
          background: var(--accent);
        }}
        .danger-fill {{
          background: var(--danger);
        }}
        .score-cell {{
          display: grid;
          grid-template-columns: 52px minmax(70px, 1fr);
          gap: 8px;
          align-items: center;
        }}
        .badge {{
          display: inline-flex;
          align-items: center;
          min-height: 24px;
          border-radius: 999px;
          padding: 3px 8px;
          font-size: 12px;
          font-weight: 800;
        }}
        .badge.attack, .badge.wrong {{
          color: var(--danger);
          background: #fee2e2;
        }}
        .badge.normal, .badge.correct {{
          color: var(--ok);
          background: #dcfce7;
        }}
        .row-id {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 34px;
          min-height: 24px;
          border-radius: 999px;
          background: #e0f2fe;
          color: #075985;
          font-weight: 800;
          font-size: 12px;
        }}
        .toolbar {{
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          align-items: center;
          margin: 8px 0 12px;
        }}
        .toolbar input {{
          width: min(360px, 100%);
          min-height: 38px;
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 8px 10px;
        }}
        .table-wrap {{
          width: 100%;
          overflow: auto;
          border: 1px solid var(--border);
          border-radius: 6px;
          margin-top: 14px;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }}
        th, td {{
          border-bottom: 1px solid var(--border);
          padding: 9px 10px;
          text-align: left;
          vertical-align: top;
        }}
        th {{
          background: #f1f5f9;
          color: #334155;
          position: sticky;
          top: 0;
        }}
        td {{
          max-width: 360px;
          overflow-wrap: anywhere;
        }}
        @media (max-width: 880px) {{
          .grid, .metrics, .chart-grid {{ grid-template-columns: 1fr; }}
          .bar-row {{ grid-template-columns: 110px 1fr 42px; }}
        }}
      </style>
      <script>
        function filterRows() {{
          const input = document.getElementById("tableFilter");
          const table = document.getElementById("predictionTable");
          if (!input || !table) return;
          const query = input.value.toLowerCase();
          for (const row of table.querySelectorAll("tbody tr")) {{
            row.style.display = row.textContent.toLowerCase().includes(query) ? "" : "none";
          }}
        }}
        function showAllRows() {{
          const table = document.getElementById("predictionTable");
          if (!table) return;
          for (const row of table.querySelectorAll("tbody tr")) row.style.display = "";
        }}
        function showOnlyWrong() {{
          const table = document.getElementById("predictionTable");
          if (!table) return;
          for (const row of table.querySelectorAll("tbody tr")) {{
            row.style.display = row.textContent.toLowerCase().includes("wrong") ? "" : "none";
          }}
        }}
        function showOnlyAttack() {{
          const table = document.getElementById("predictionTable");
          if (!table) return;
          for (const row of table.querySelectorAll("tbody tr")) {{
            row.style.display = row.textContent.toLowerCase().includes("attack") ? "" : "none";
          }}
        }}
      </script>
    </head>
    <body>
      <main>
        <h1>Guardrail Dashboard</h1>
        <p class="subhead">Upload a CSV, choose the text and optional label columns, then run the guardrail classifier.</p>
        {error_html}
        <section>
          <form action="/dashboard" method="post" enctype="multipart/form-data">
            <div class="grid">
              <div>
                <label for="file">CSV file</label>
                <input id="file" name="file" type="file" accept=".csv,text/csv" required>
              </div>
              <div>
                <label for="text_column">Text column</label>
                <input id="text_column" name="text_column" type="text" value="text" required>
              </div>
              <div>
                <label for="label_column">Label column</label>
                <input id="label_column" name="label_column" type="text" value="label">
              </div>
              <div>
                <label for="threshold">Attack threshold</label>
                <input id="threshold" name="threshold" type="number" min="0" max="1" step="0.01" value="{threshold}">
              </div>
              <div>
                <label for="max_length">Max length</label>
                <input id="max_length" name="max_length" type="number" min="8" step="1" value="{max_length}">
              </div>
              <div>
                <label for="preview_rows">Preview rows</label>
                <input id="preview_rows" name="preview_rows" type="number" min="1" max="500" step="1" value="50">
              </div>
              <div>
                <label for="batch_size">Batch size</label>
                <input id="batch_size" name="batch_size" type="number" min="1" max="256" step="1" value="{DEFAULT_DASHBOARD_BATCH_SIZE}">
              </div>
              <div>
                <label>Validation</label>
                <label class="checkbox">
                  <input name="validate_binary" type="checkbox" value="true" checked>
                  Require binary labels
                </label>
              </div>
            </div>
            <div class="actions">
              <button type="submit">Run prediction</button>
              <span class="hint">Legacy files can use text column <strong>Instruct</strong> and label column <strong>Label</strong>.</span>
            </div>
          </form>
        </section>
        {result_html}
      </main>
    </body>
    </html>
    """


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


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


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(render_dashboard())


@app.post("/dashboard", response_class=HTMLResponse)
async def dashboard_upload(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
    validate_binary: bool = Form(False),
    threshold: str | None = Form(None),
    max_length: str | None = Form(None),
    preview_rows: int = Form(50),
    batch_size: int = Form(DEFAULT_DASHBOARD_BATCH_SIZE),
) -> HTMLResponse:
    try:
        if not file.filename:
            raise ValueError("Please upload a CSV file.")
        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        content = await file.read()
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")

        text_column = text_column.strip()
        label_column = label_column.strip()
        if text_column not in frame.columns:
            raise ValueError(
                f"Text column '{text_column}' was not found. Available columns: "
                f"{', '.join(map(str, frame.columns))}"
            )

        has_label = bool(label_column) and label_column in frame.columns
        if label_column and label_column not in frame.columns:
            raise ValueError(
                f"Label column '{label_column}' was not found. Leave it blank for prediction-only mode."
            )

        work = frame.copy()
        work[text_column] = work[text_column].astype(str).str.strip()
        work = work[work[text_column].ne("")].reset_index(drop=True)
        if work.empty:
            raise ValueError("No non-empty text rows found after cleaning.")

        if preview_rows < 1:
            raise ValueError("preview_rows must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        requested_threshold = parse_optional_float(threshold)
        requested_max_length = parse_optional_int(max_length)

        true_labels: list[int] | None = None
        if has_label:
            numeric_labels = pd.to_numeric(work[label_column], errors="coerce")
            invalid_count = int(numeric_labels.isna().sum())
            if invalid_count:
                raise ValueError(f"Label column has {invalid_count} non-numeric value(s).")

            true_labels = numeric_labels.astype(int).tolist()
            unique_labels = sorted(set(true_labels))
            if validate_binary and not set(unique_labels).issubset({0, 1}):
                raise ValueError(
                    "Label column is not binary. Expected only 0/1 values; "
                    f"found {unique_labels}."
                )

        predictions = predict_texts_batched(
            work[text_column].tolist(),
            max_length=requested_max_length,
            threshold=requested_threshold,
            batch_size=batch_size,
        )

        output = work.copy()
        output.insert(0, "prediction_no", range(1, len(output) + 1))
        output["predicted_label"] = [prediction.label for prediction in predictions]
        output["predicted_label_id"] = [prediction.label_id for prediction in predictions]
        output["predicted_score"] = [prediction.score for prediction in predictions]
        output["attack_score"] = [prediction.attack_score for prediction in predictions]
        output["threshold"] = [prediction.threshold for prediction in predictions]
        output["is_attack"] = [prediction.is_attack for prediction in predictions]
        output["text_length"] = output[text_column].astype(str).str.len()

        metrics: dict[str, Any] = {
            "rows": len(output),
            "attack_rate": f"{output['is_attack'].mean():.3f}",
            "threshold": f"{predictions[0].threshold:.2f}",
            "avg_attack_score": f"{output['attack_score'].mean():.3f}",
            "filename": file.filename,
        }

        wrong_rows = pd.DataFrame()
        if true_labels is not None:
            output["true_label"] = true_labels
            output["true_is_attack"] = output["true_label"].astype(int) == 1
            output["is_wrong"] = output["true_is_attack"] != output["is_attack"]
            wrong_rows = output[output["is_wrong"]].copy()
            accuracy = 1 - (len(wrong_rows) / max(1, len(output)))
            metrics["accuracy"] = f"{accuracy:.3f}"
            metrics["wrong"] = len(wrong_rows)
            metrics["wrong_rate"] = f"{output['is_wrong'].mean():.3f}"

        download_id = uuid.uuid4().hex
        DASHBOARD_DOWNLOADS[download_id] = output.to_csv(index=False)

        label_distribution = (
            make_distribution(output[label_column])
            if has_label and label_column in output.columns
            else []
        )
        category_distribution = (
            make_distribution(output["category"])
            if "category" in output.columns
            else []
        )
        risky_columns = [
            column
            for column in [
                "prediction_no",
                text_column,
                label_column if has_label else None,
                "predicted_label",
                "attack_score",
                "threshold",
                "is_attack",
                "is_wrong" if "is_wrong" in output.columns else None,
            ]
            if column
        ]
        risky_rows = (
            output.sort_values("attack_score", ascending=False)
            .head(25)[risky_columns]
            .to_dict(orient="records")
        )
        low_confidence_rows = (
            output.sort_values("predicted_score", ascending=True)
            .head(25)[risky_columns]
            .to_dict(orient="records")
        )

        preview_columns = [
            column
            for column in [
                "prediction_no",
                text_column,
                label_column if has_label else None,
                "predicted_label",
                "attack_score",
                "threshold",
                "is_attack",
                "is_wrong" if "is_wrong" in output.columns else None,
            ]
            if column
        ]
        wrong_columns = [
            column
            for column in [
                "prediction_no",
                text_column,
                label_column if has_label else None,
                "predicted_label",
                "attack_score",
                "threshold",
                "is_attack",
            ]
            if column
        ]
        result = {
            "metrics": metrics,
            "download_id": download_id,
            "column_profiles": profile_columns(
                frame,
                text_column=text_column,
                label_column=label_column if has_label else None,
            ),
            "prediction_distribution": make_distribution(output["predicted_label"]),
            "attack_distribution": make_distribution(output["is_attack"].map({True: "attack", False: "normal"})),
            "label_distribution": label_distribution,
            "category_distribution": category_distribution,
            "confidence_distribution": make_numeric_bins(
                output["predicted_score"],
                [
                    ("0.00-0.50", 0, 0.5),
                    ("0.50-0.70", 0.5, 0.7),
                    ("0.70-0.85", 0.7, 0.85),
                    ("0.85-0.95", 0.85, 0.95),
                    ("0.95-1.00", 0.95, 1.01),
                ],
            ),
            "text_length_distribution": make_text_length_bins(output["text_length"]),
            "threshold_curve": make_threshold_curve(output),
            "confusion_distribution": make_confusion_counts(output),
            "wrong_by_category": group_wrong_distribution(output, "category"),
            "wrong_by_source_file": group_wrong_distribution(output, "source_file"),
            "risky_columns": risky_columns,
            "risky_rows": risky_rows,
            "low_confidence_rows": low_confidence_rows,
            "columns": preview_columns,
            "preview_rows": output[preview_columns]
            .head(min(preview_rows, 500))
            .to_dict(orient="records"),
            "wrong_columns": wrong_columns,
            "wrong_rows": wrong_rows[wrong_columns].head(100).to_dict(orient="records")
            if not wrong_rows.empty
            else [],
        }
        return HTMLResponse(render_dashboard(result=result))
    except Exception as exc:
        return HTMLResponse(render_dashboard(error=str(exc)), status_code=400)


@app.get("/dashboard/download/{download_id}")
def dashboard_download(download_id: str) -> StreamingResponse:
    csv_text = DASHBOARD_DOWNLOADS.get(download_id)
    if csv_text is None:
        raise HTTPException(status_code=404, detail="Download not found or expired.")

    return StreamingResponse(
        io.StringIO(csv_text),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=guardrail_predictions.csv"},
    )


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    return predict_texts(
        [request.text],
        max_length=request.max_length,
        threshold=request.threshold,
    )[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchPredictRequest) -> list[Prediction]:
    return predict_texts(
        request.texts,
        max_length=request.max_length,
        threshold=request.threshold,
    )
