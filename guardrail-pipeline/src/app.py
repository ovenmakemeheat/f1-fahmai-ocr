from __future__ import annotations

import html
import io
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.dashboard_utils import prediction_row, prediction_rows
from src.guardrail_model import (
    DEFAULT_ATTACK_THRESHOLD,
    DEFAULT_DASHBOARD_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    Prediction,
    load_model,
    predict_texts,
    predict_texts_batched,
    resolve_model_id,
)
from src.llm_guardrail_model import predict_text_with_llm

DASHBOARD_DOWNLOADS: dict[str, str] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(
    title="Guardrail Pipeline API",
    description="Serves the finetuned WangchanBERT guardrail classifier.",
    version="0.1.0",
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int | None = None
    threshold: float | None = None


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    max_length: int | None = None
    threshold: float | None = None


class ResponseScore(BaseModel):
    is_attack: bool
    score: float


class PredictResponse(BaseModel):
    is_attack: bool
    message: str
    text: str
    threshold: float
    confident: float
    scores: list[ResponseScore]


class HealthResponse(BaseModel):
    status: str
    default_model: str
    model_id: str
    device: str
    loaded: bool


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def format_float(value: float) -> str:
    return f"{value:.3f}"


def prediction_score_by_attack(prediction: Prediction, is_attack: bool) -> float:
    target_labels = {"1", "attack", "prompt_injection"} if is_attack else {"0", "normal"}
    for score in prediction.scores:
        if str(score.label).lower() in target_labels:
            return score.score
    return prediction.attack_score if is_attack else max(0.0, 1.0 - prediction.attack_score)


def prediction_response(prediction: Prediction) -> PredictResponse:
    normal_score = prediction_score_by_attack(prediction, is_attack=False)
    attack_score = prediction_score_by_attack(prediction, is_attack=True)
    return PredictResponse(
        is_attack=prediction.is_attack,
        message=prediction.message,
        text=prediction.text,
        threshold=prediction.threshold,
        confident=attack_score if prediction.is_attack else normal_score,
        scores=[
            ResponseScore(is_attack=False, score=normal_score),
            ResponseScore(is_attack=True, score=attack_score),
        ],
    )


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


def render_dashboard(
    error: str | None = None,
    result: dict[str, Any] | None = None,
    title: str = "Guardrail Dashboard",
    subtitle: str = "Upload a CSV, choose the text and optional label columns, then run the guardrail classifier.",
    query_action: str = "/dashboard/query",
    upload_action: str = "/dashboard",
    api_endpoint: str = "/predict",
    default_max_length: int = DEFAULT_MAX_LENGTH,
    max_length_env: str = "GUARDRAIL_MAX_LENGTH",
) -> str:
    threshold = escape(os.getenv("GUARDRAIL_ATTACK_THRESHOLD", str(DEFAULT_ATTACK_THRESHOLD)))
    max_length = escape(os.getenv(max_length_env, str(default_max_length)))
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
      <title>{escape(title)}</title>
      <style>
        @font-face {{
          font-family: "LINE Seed Sans TH";
          src: url("/static/fonts/LINESeedSansTH_Rg.ttf") format("truetype");
          font-weight: 400;
          font-style: normal;
          font-display: swap;
        }}
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
          font-family: "LINE Seed Sans TH", Tahoma, sans-serif;
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
        textarea {{
          width: 100%;
          min-height: 180px;
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 12px;
          background: #fff;
          color: var(--text);
          font: 14px/1.5 "LINE Seed Sans TH", Tahoma, sans-serif;
          resize: vertical;
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
        .field-note {{
          margin-top: 6px;
          color: var(--muted);
          font-size: 12px;
        }}
        .column-panel {{
          display: none;
          border: 1px solid var(--border);
          border-radius: 6px;
          background: #f8fafc;
          padding: 12px;
          margin-top: 14px;
        }}
        .column-panel.active {{
          display: block;
        }}
        .column-panel-head {{
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: center;
          margin-bottom: 10px;
        }}
        .column-panel-title {{
          font-size: 13px;
          font-weight: 800;
        }}
        .column-panel-status {{
          color: var(--muted);
          font-size: 12px;
        }}
        .column-picker {{
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }}
        .column-presets {{
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }}
        .preset-button {{
          border: 1px solid var(--border);
          border-radius: 6px;
          background: #fff;
          color: #334155;
          min-height: 34px;
          padding: 7px 10px;
          font-size: 12px;
          font-weight: 800;
        }}
        .preset-button:hover {{
          border-color: var(--accent);
          background: #ecfdf5;
          color: var(--accent-dark);
        }}
        .preset-button:disabled {{
          opacity: 0.45;
          cursor: not-allowed;
        }}
        .column-mode {{
          display: inline-flex;
          gap: 6px;
          padding: 4px;
          border: 1px solid var(--border);
          border-radius: 6px;
          background: #fff;
        }}
        .mode-button {{
          border: 1px solid transparent;
          border-radius: 5px;
          background: transparent;
          color: #334155;
          min-height: 32px;
          padding: 6px 10px;
          font-size: 12px;
        }}
        .mode-button.active {{
          background: var(--accent);
          color: #fff;
        }}
        .column-preview-wrap {{
          margin-top: 12px;
          border: 1px solid var(--border);
          border-radius: 6px;
          background: #fff;
          overflow: auto;
        }}
        .column-preview {{
          min-width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }}
        .column-preview th,
        .column-preview td {{
          border-bottom: 1px solid var(--border);
          border-right: 1px solid var(--border);
          padding: 7px 8px;
          max-width: 260px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }}
        .column-preview th {{
          background: #f1f5f9;
          position: sticky;
          top: 0;
          z-index: 1;
        }}
        .column-header-button {{
          width: 100%;
          min-height: 30px;
          border: 1px solid var(--border);
          border-radius: 5px;
          background: #fff;
          color: #334155;
          text-align: left;
          padding: 6px 8px;
          font-size: 12px;
          font-weight: 900;
        }}
        .column-header-button:hover {{
          border-color: var(--accent);
          background: #ecfdf5;
          color: var(--accent-dark);
        }}
        .column-header-button.selected-text {{
          border-color: var(--accent);
          background: #ccfbf1;
          color: var(--accent-dark);
        }}
        .column-header-button.selected-label {{
          border-color: #0369a1;
          background: #e0f2fe;
          color: #075985;
        }}
        .selected-column {{
          min-height: 38px;
          border: 1px dashed var(--border);
          border-radius: 6px;
          background: #f8fafc;
          color: var(--muted);
          display: flex;
          align-items: center;
          padding: 8px 10px;
          font-size: 13px;
          overflow-wrap: anywhere;
        }}
        .selected-column.filled {{
          border-style: solid;
          background: #fff;
          color: var(--text);
          font-weight: 800;
        }}
        .column-group {{
          display: inline-flex;
          align-items: center;
          gap: 4px;
          border: 1px solid var(--border);
          border-radius: 999px;
          background: #fff;
          padding: 3px;
        }}
        .column-name {{
          color: #334155;
          font-size: 12px;
          font-weight: 800;
          padding: 0 8px;
          max-width: 220px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }}
        .column-chip {{
          border: 1px solid var(--border);
          border-radius: 999px;
          background: #fff;
          color: #334155;
          min-height: 32px;
          padding: 6px 10px;
          font-size: 12px;
          font-weight: 700;
          cursor: pointer;
        }}
        .column-chip:hover {{
          border-color: var(--accent);
          color: var(--accent-dark);
          background: #ecfdf5;
        }}
        .column-chip.selected-text {{
          border-color: var(--accent);
          background: #ccfbf1;
          color: var(--accent-dark);
        }}
        .column-chip.selected-label {{
          border-color: #0369a1;
          background: #e0f2fe;
          color: #075985;
        }}
        .submit-button {{
          min-width: 132px;
        }}
        body.loading .submit-button {{
          background: #64748b;
        }}
        body.loading .submit-button::after {{
          content: "";
          display: inline-block;
          width: 12px;
          height: 12px;
          margin-left: 8px;
          border: 2px solid rgba(255, 255, 255, 0.55);
          border-top-color: #fff;
          border-radius: 999px;
          vertical-align: -2px;
          animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{
          to {{ transform: rotate(360deg); }}
        }}
        body.loading {{
          cursor: progress;
        }}
        body.loading button, body.loading input, body.loading textarea {{
          pointer-events: none;
        }}
        .loading-overlay {{
          display: none;
          position: fixed;
          inset: 0;
          z-index: 40;
          background: rgba(15, 23, 42, 0.42);
          align-items: center;
          justify-content: center;
          padding: 20px;
        }}
        body.loading .loading-overlay {{
          display: flex;
        }}
        .loading-card {{
          width: min(420px, 100%);
          border-radius: 8px;
          background: #fff;
          border: 1px solid var(--border);
          box-shadow: 0 24px 80px rgba(15, 23, 42, 0.24);
          padding: 20px;
        }}
        .loading-title {{
          font-weight: 900;
          margin-bottom: 8px;
        }}
        .loading-text {{
          color: var(--muted);
          font-size: 13px;
          line-height: 1.45;
        }}
        .loading-bar {{
          height: 8px;
          border-radius: 999px;
          overflow: hidden;
          background: #e2e8f0;
          margin-top: 16px;
        }}
        .loading-bar::before {{
          content: "";
          display: block;
          width: 38%;
          height: 100%;
          background: var(--accent);
          animation: loadingSweep 1.1s ease-in-out infinite;
        }}
        @keyframes loadingSweep {{
          0% {{ transform: translateX(-110%); }}
          100% {{ transform: translateX(280%); }}
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
        let activeColumnTarget = "text";
        function parseCsvLine(line) {{
          const cells = [];
          let current = "";
          let quoted = false;
          for (let index = 0; index < line.length; index++) {{
            const char = line[index];
            const next = line[index + 1];
            if (char === '"' && quoted && next === '"') {{
              current += '"';
              index++;
            }} else if (char === '"') {{
              quoted = !quoted;
            }} else if (char === "," && !quoted) {{
              cells.push(current.trim().replace(/^\uFEFF/, ""));
              current = "";
            }} else {{
              current += char;
            }}
          }}
          cells.push(current.trim().replace(/^\uFEFF/, ""));
          return cells;
        }}
        function parseCsvPreview(text) {{
          const lines = String(text || "").split(/\r?\n/).filter((line) => line.trim());
          if (!lines.length) return {{ columns: [], rows: [] }};
          const columns = parseCsvLine(lines[0]).filter(Boolean);
          const rows = lines.slice(1, 7).map((line) => parseCsvLine(line));
          return {{ columns, rows }};
        }}
        function refreshColumnChipState() {{
          const textColumn = document.getElementById("text_column")?.value;
          const labelColumn = document.getElementById("label_column")?.value;
          const textDisplay = document.getElementById("selectedTextColumn");
          const labelDisplay = document.getElementById("selectedLabelColumn");
          if (textDisplay) {{
            textDisplay.textContent = textColumn || "Click Text on a detected column";
            textDisplay.classList.toggle("filled", Boolean(textColumn));
          }}
          if (labelDisplay) {{
            labelDisplay.textContent = labelColumn || "Optional: click Label on a 0/1 column";
            labelDisplay.classList.toggle("filled", Boolean(labelColumn));
          }}
          for (const chip of document.querySelectorAll(".column-chip, .column-header-button")) {{
            chip.classList.toggle("selected-text", chip.dataset.column === textColumn);
            chip.classList.toggle("selected-label", chip.dataset.column === labelColumn);
          }}
          for (const button of document.querySelectorAll(".mode-button")) {{
            button.classList.toggle("active", button.dataset.target === activeColumnTarget);
          }}
        }}
        function chooseColumn(column, target) {{
          const input = document.getElementById(target === "label" ? "label_column" : "text_column");
          if (!input) return;
          input.value = column;
          refreshColumnChipState();
        }}
        function applyColumnPreset(textColumn, labelColumn = "") {{
          const textInput = document.getElementById("text_column");
          const labelInput = document.getElementById("label_column");
          if (textInput) textInput.value = textColumn;
          if (labelInput) labelInput.value = labelColumn;
          refreshColumnChipState();
        }}
        function findColumn(columns, name) {{
          return columns.find((column) => column.toLowerCase() === name.toLowerCase()) || "";
        }}
        function setColumnTarget(target) {{
          activeColumnTarget = target === "label" ? "label" : "text";
          refreshColumnChipState();
        }}
        function renderColumnPicker(columns, rows = []) {{
          const panel = document.getElementById("columnPanel");
          const picker = document.getElementById("columnPicker");
          const status = document.getElementById("columnStatus");
          if (!panel || !picker || !status) return;
          if (!columns.length) {{
            panel.classList.remove("active");
            picker.innerHTML = "";
            status.textContent = "";
            return;
          }}
          panel.classList.add("active");
          status.textContent = `${{columns.length}} columns detected`;
          const textInput = document.getElementById("text_column");
          const labelInput = document.getElementById("label_column");
          if (textInput) textInput.value = "";
          if (labelInput) labelInput.value = "";
          picker.replaceChildren();
          const mode = document.createElement("div");
          mode.className = "column-mode";
          for (const target of ["text", "label"]) {{
            const button = document.createElement("button");
            button.type = "button";
            button.className = "mode-button";
            button.dataset.target = target;
            button.textContent = target === "text" ? "Assign as Text" : "Assign as Label";
            button.addEventListener("click", () => setColumnTarget(target));
            mode.appendChild(button);
          }}
          const presets = document.createElement("div");
          presets.className = "column-presets";
          const presetConfigs = [
            {{
              label: "Preset: questions_formatted_id",
              text: findColumn(columns, "Instruct"),
              labelColumn: findColumn(columns, "Label"),
              requireLabel: true,
            }},
            {{
              label: "Preset: INJ",
              text: findColumn(columns, "question"),
              labelColumn: findColumn(columns, "Label"),
              requireLabel: true,
            }},
            {{
              label: "Preset: redteam question",
              text: findColumn(columns, "question"),
              labelColumn: "",
              requireLabel: false,
            }},
          ];
          for (const config of presetConfigs) {{
            const button = document.createElement("button");
            button.type = "button";
            button.className = "preset-button";
            button.textContent = config.label;
            button.disabled = !config.text || (config.requireLabel && !config.labelColumn);
            button.addEventListener("click", () => applyColumnPreset(config.text, config.labelColumn));
            presets.appendChild(button);
          }}
          const previewWrap = document.createElement("div");
          previewWrap.className = "column-preview-wrap";
          const table = document.createElement("table");
          table.className = "column-preview";
          const thead = document.createElement("thead");
          const headerRow = document.createElement("tr");
          for (const column of columns) {{
            const th = document.createElement("th");
            const button = document.createElement("button");
            button.type = "button";
            button.className = "column-header-button";
            button.dataset.column = column;
            button.textContent = column;
            button.title = `Click to assign ${{column}} as ${{activeColumnTarget}}`;
            button.addEventListener("click", () => chooseColumn(column, activeColumnTarget));
            th.appendChild(button);
            headerRow.appendChild(th);
          }}
          thead.appendChild(headerRow);
          table.appendChild(thead);
          const tbody = document.createElement("tbody");
          for (const row of rows) {{
            const tr = document.createElement("tr");
            for (let index = 0; index < columns.length; index++) {{
              const td = document.createElement("td");
              td.textContent = row[index] || "";
              td.title = row[index] || "";
              tr.appendChild(td);
            }}
            tbody.appendChild(tr);
          }}
          table.appendChild(tbody);
          previewWrap.appendChild(table);
          picker.append(presets, mode, previewWrap);
          refreshColumnChipState();
        }}
        function loadCsvColumns(input) {{
          const file = input.files && input.files[0];
          if (!file) {{
            renderColumnPicker([]);
            return;
          }}
          const reader = new FileReader();
          reader.onload = () => {{
            const preview = parseCsvPreview(reader.result || "");
            renderColumnPicker(preview.columns, preview.rows);
          }};
          reader.onerror = () => renderColumnPicker([]);
          reader.readAsText(file.slice(0, 65536));
        }}
        function startLoading(message) {{
          const overlayText = document.getElementById("loadingText");
          if (overlayText) overlayText.textContent = message;
          document.body.classList.add("loading");
          for (const button of document.querySelectorAll("button[type='submit']")) {{
            button.setAttribute("aria-busy", "true");
          }}
        }}
        window.addEventListener("DOMContentLoaded", () => {{
          for (const inputId of ["text_column", "label_column"]) {{
            const input = document.getElementById(inputId);
            if (input) input.addEventListener("input", refreshColumnChipState);
          }}
          const fileInput = document.getElementById("file");
          if (fileInput) fileInput.addEventListener("change", () => loadCsvColumns(fileInput));
          refreshColumnChipState();
          for (const form of document.querySelectorAll("form")) {{
            form.addEventListener("submit", () => {{
              const isUpload = form.enctype === "multipart/form-data";
              startLoading(isUpload ? "Processing CSV rows with the guardrail model..." : "Classifying the request...");
            }});
          }}
        }});
      </script>
    </head>
    <body>
      <main>
        <h1>{escape(title)}</h1>
        <p class="subhead">{escape(subtitle)}</p>
        {error_html}
        <section>
          <div class="section-title">Single Text Query</div>
          <form action="{escape(query_action)}" method="post">
            <label for="query_text">Text editor</label>
            <textarea id="query_text" name="query_text" placeholder="Paste or type one request to classify..." required></textarea>
            <div class="grid" style="margin-top: 14px;">
              <div>
                <label for="query_threshold">Attack threshold</label>
                <input id="query_threshold" name="threshold" type="number" min="0" max="1" step="0.01" value="{threshold}">
              </div>
              <div>
                <label for="query_max_length">Max length</label>
                <input id="query_max_length" name="max_length" type="number" min="8" step="1" value="{max_length}">
              </div>
            </div>
            <div class="actions">
              <button class="submit-button" type="submit">Classify text</button>
              <span class="hint">Use this for quick manual checks before uploading a full CSV.</span>
            </div>
          </form>
        </section>
        <section>
          <div class="section-title">CSV Batch Prediction</div>
          <form action="{escape(upload_action)}" method="post" enctype="multipart/form-data">
            <div class="grid">
              <div>
                <label for="file">CSV file</label>
                <input id="file" name="file" type="file" accept=".csv,text/csv" onchange="loadCsvColumns(this)" required>
                <div class="field-note">Choose a file, then use the preview headers below.</div>
              </div>
              <div>
                <label for="text_column">Text column</label>
                <input id="text_column" name="text_column" type="hidden" value="">
                <div id="selectedTextColumn" class="selected-column"></div>
              </div>
              <div>
                <label for="label_column">Label column</label>
                <input id="label_column" name="label_column" type="hidden" value="">
                <div id="selectedLabelColumn" class="selected-column"></div>
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
            <div id="columnPanel" class="column-panel">
              <div class="column-panel-head">
                <div class="column-panel-title">Column Selection</div>
                <div id="columnStatus" class="column-panel-status"></div>
              </div>
              <div id="columnPicker" class="column-picker"></div>
              <div class="field-note">Use a preset for dataset/test CSVs, or select a mode and click a preview header.</div>
            </div>
            <div class="actions">
              <button class="submit-button" type="submit">Run prediction</button>
              <span class="hint">API endpoint: <strong>{escape(api_endpoint)}</strong>. Legacy files can use text column <strong>Instruct</strong> and label column <strong>Label</strong>.</span>
            </div>
          </form>
        </section>
        {result_html}
      </main>
      <div class="loading-overlay" aria-live="polite" aria-busy="true">
        <div class="loading-card">
          <div class="loading-title">Running Guardrail</div>
          <div id="loadingText" class="loading-text">Preparing request...</div>
          <div class="loading-bar"></div>
        </div>
      </div>
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


def render_dashboard_v2(
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> str:
    return render_dashboard(
        error=error,
        result=result,
        title="Guardrail Dashboard V2",
        subtitle=(
            "Run the Qwen3 LoRA guardrail. Predictions use label-token probabilities; "
            "label 1 probability is compared with the threshold."
        ),
        query_action="/dashboardv2/query",
        upload_action="/dashboardv2",
        api_endpoint="/predictv2",
        default_max_length=2048,
        max_length_env="GUARDRAIL_LLM_MAX_LENGTH",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_id = resolve_model_id(DEFAULT_MODEL_NAME)
    loaded = load_model.cache_info().currsize > 0
    device = os.getenv("GUARDRAIL_DEVICE", "auto")
    return HealthResponse(
        status="ok",
        default_model=DEFAULT_MODEL_NAME,
        model_id=model_id,
        device=device,
        loaded=loaded,
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(render_dashboard())


@app.get("/dashboardv2", response_class=HTMLResponse)
@app.get("/dashboard/predictv2", response_class=HTMLResponse)
def dashboard_v2() -> HTMLResponse:
    return HTMLResponse(render_dashboard_v2())


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
        if not text_column:
            raise ValueError("Please select a text column from the CSV preview.")
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
            model_name=DEFAULT_MODEL_NAME,
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


@app.post("/dashboardv2", response_class=HTMLResponse)
@app.post("/dashboard/predictv2", response_class=HTMLResponse)
async def dashboard_v2_upload(
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
        if not text_column:
            raise ValueError("Please select a text column from the CSV preview.")
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

        predictions = [
            predict_text_with_llm(
                text,
                model_name=DEFAULT_MODEL_NAME,
                max_length=requested_max_length,
                threshold=requested_threshold,
            )
            for text in work[text_column].tolist()
        ]

        output = work.copy()
        for column, values in pd.DataFrame(
            prediction_rows(predictions, work[text_column].tolist(), text_column)
        ).items():
            if column != text_column:
                output[column] = values
        output["text_length"] = output[text_column].astype(str).str.len()

        metrics: dict[str, Any] = {
            "endpoint": "/predictv2",
            "rows": len(output),
            "model": "qwen3-4b-fahmai-guardrails-v2",
            "scoring": "next-token label probability",
            "threshold": f"{predictions[0].threshold:.2f}",
            "max_length": requested_max_length or 2048,
            "attack_rate": f"{output['is_attack'].mean():.3f}",
            "avg_label_1_probability": f"{output['label_1_probability'].mean():.3f}",
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
        preview_columns = [
            column
            for column in [
                "prediction_no",
                text_column,
                label_column if has_label else None,
                "predicted_label",
                "label_0_probability",
                "label_1_probability",
                "label_confidence",
                "threshold",
                "is_attack",
                "message",
                "is_wrong" if "is_wrong" in output.columns else None,
            ]
            if column
        ]
        risky_columns = preview_columns
        wrong_columns = [
            column
            for column in [
                "prediction_no",
                text_column,
                label_column if has_label else None,
                "predicted_label",
                "label_1_probability",
                "threshold",
                "is_attack",
                "message",
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
            "category_distribution": make_distribution(output["Category"])
            if "Category" in output.columns
            else [],
            "confidence_distribution": make_numeric_bins(
                output["label_confidence"],
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
            "wrong_by_category": group_wrong_distribution(output, "Category"),
            "wrong_by_source_file": group_wrong_distribution(output, "source_file"),
            "risky_columns": risky_columns,
            "risky_rows": output.sort_values("label_1_probability", ascending=False)
            .head(25)[risky_columns]
            .to_dict(orient="records"),
            "low_confidence_rows": output.sort_values("label_confidence", ascending=True)
            .head(25)[risky_columns]
            .to_dict(orient="records"),
            "columns": preview_columns,
            "preview_rows": output[preview_columns]
            .head(min(preview_rows, 500))
            .to_dict(orient="records"),
            "wrong_columns": wrong_columns,
            "wrong_rows": wrong_rows[wrong_columns].head(100).to_dict(orient="records")
            if not wrong_rows.empty
            else [],
        }
        return HTMLResponse(render_dashboard_v2(result=result))
    except Exception as exc:
        return HTMLResponse(render_dashboard_v2(error=str(exc)), status_code=400)


@app.post("/dashboard/query", response_class=HTMLResponse)
async def dashboard_query(
    query_text: str = Form(...),
    threshold: str | None = Form(None),
    max_length: str | None = Form(None),
) -> HTMLResponse:
    try:
        text = query_text.strip()
        if not text:
            raise ValueError("Text query must not be empty.")

        requested_threshold = parse_optional_float(threshold)
        requested_max_length = parse_optional_int(max_length)
        prediction = predict_texts(
            [text],
            model_name=DEFAULT_MODEL_NAME,
            max_length=requested_max_length,
            threshold=requested_threshold,
        )[0]

        output = pd.DataFrame(
            [
                {
                    "prediction_no": 1,
                    "text": text,
                    "predicted_label": prediction.label,
                    "predicted_label_id": prediction.label_id,
                    "predicted_score": prediction.score,
                    "attack_score": prediction.attack_score,
                    "threshold": prediction.threshold,
                    "is_attack": prediction.is_attack,
                    "text_length": len(text),
                }
            ]
        )

        download_id = uuid.uuid4().hex
        DASHBOARD_DOWNLOADS[download_id] = output.to_csv(index=False)
        preview_columns = [
            "prediction_no",
            "text",
            "predicted_label",
            "attack_score",
            "threshold",
            "is_attack",
        ]

        result = {
            "metrics": {
                "rows": 1,
                "decision": "attack" if prediction.is_attack else "normal",
                "attack_score": format_float(prediction.attack_score),
                "threshold": format_float(prediction.threshold),
                "predicted_label": prediction.label,
            },
            "download_id": download_id,
            "column_profiles": profile_columns(output, text_column="text", label_column=None),
            "prediction_distribution": make_distribution(output["predicted_label"]),
            "attack_distribution": make_distribution(output["is_attack"].map({True: "attack", False: "normal"})),
            "label_distribution": [],
            "category_distribution": [],
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
            "confusion_distribution": [],
            "wrong_by_category": [],
            "wrong_by_source_file": [],
            "risky_columns": preview_columns,
            "risky_rows": output[preview_columns].to_dict(orient="records"),
            "low_confidence_rows": output[preview_columns].to_dict(orient="records"),
            "columns": preview_columns,
            "preview_rows": output[preview_columns].to_dict(orient="records"),
            "wrong_columns": preview_columns,
            "wrong_rows": [],
        }
        return HTMLResponse(render_dashboard(result=result))
    except Exception as exc:
        return HTMLResponse(render_dashboard(error=str(exc)), status_code=400)


@app.post("/dashboardv2/query", response_class=HTMLResponse)
@app.post("/dashboard/predictv2/query", response_class=HTMLResponse)
async def dashboard_v2_query(
    query_text: str = Form(...),
    threshold: str | None = Form(None),
    max_length: str | None = Form(None),
) -> HTMLResponse:
    try:
        text = query_text.strip()
        if not text:
            raise ValueError("Text query must not be empty.")

        requested_threshold = parse_optional_float(threshold)
        requested_max_length = parse_optional_int(max_length)
        prediction = predict_text_with_llm(
            text,
            model_name=DEFAULT_MODEL_NAME,
            max_length=requested_max_length,
            threshold=requested_threshold,
        )

        output = pd.DataFrame([prediction_row(prediction, 1, text)])
        output["text_length"] = output["text"].astype(str).str.len()

        download_id = uuid.uuid4().hex
        DASHBOARD_DOWNLOADS[download_id] = output.to_csv(index=False)
        preview_columns = [
            "prediction_no",
            "text",
            "predicted_label",
            "label_0_probability",
            "label_1_probability",
            "label_confidence",
            "threshold",
            "is_attack",
            "message",
        ]

        result = {
            "metrics": {
                "endpoint": "/predictv2",
                "rows": 1,
                "model": "qwen3-4b-fahmai-guardrails-v2",
                "scoring": "next-token label probability",
                "decision": "attack" if prediction.is_attack else "normal",
                "label_1_probability": format_float(prediction.attack_score),
                "threshold": format_float(prediction.threshold),
                "message": prediction.message,
            },
            "download_id": download_id,
            "column_profiles": profile_columns(output, text_column="text", label_column=None),
            "prediction_distribution": make_distribution(output["predicted_label"]),
            "attack_distribution": make_distribution(output["is_attack"].map({True: "attack", False: "normal"})),
            "label_distribution": [],
            "category_distribution": [],
            "confidence_distribution": make_numeric_bins(
                output["label_confidence"],
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
            "confusion_distribution": [],
            "wrong_by_category": [],
            "wrong_by_source_file": [],
            "risky_columns": preview_columns,
            "risky_rows": output[preview_columns].to_dict(orient="records"),
            "low_confidence_rows": output[preview_columns].to_dict(orient="records"),
            "columns": preview_columns,
            "preview_rows": output[preview_columns].to_dict(orient="records"),
            "wrong_columns": preview_columns,
            "wrong_rows": [],
        }
        return HTMLResponse(render_dashboard_v2(result=result))
    except Exception as exc:
        return HTMLResponse(render_dashboard_v2(error=str(exc)), status_code=400)


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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    prediction = predict_texts(
        [request.text],
        model_name=DEFAULT_MODEL_NAME,
        max_length=request.max_length,
        threshold=request.threshold,
    )[0]
    return prediction_response(prediction)


@app.post("/predictv2", response_model=PredictResponse)
def predict_llm(request: PredictRequest) -> PredictResponse:
    prediction = predict_text_with_llm(
        request.text,
        model_name=DEFAULT_MODEL_NAME,
        max_length=request.max_length,
        threshold=request.threshold,
    )
    return prediction_response(prediction)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest) -> list[PredictResponse]:
    predictions = predict_texts(
        request.texts,
        model_name=DEFAULT_MODEL_NAME,
        max_length=request.max_length,
        threshold=request.threshold,
    )
    return [prediction_response(prediction) for prediction in predictions]
