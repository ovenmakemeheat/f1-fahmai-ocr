# Finetuning Pipeline Implementation Plan

## Goal

Create a reproducible Jupyter notebook pipeline for finetuning Thai/English text classifiers on the synthetic enterprise RAG guardrail dataset.

The pipeline will use `wangchanberta-base-att-spm-uncased` as the default model and also support RoBERTa and PhayaThaiBERT model families through a single model registry.

## Input Data

Expected CSV path, relative to the `guardrail-pipeline` project root:

```text
dataset/fahmai_guardrail_bert_all.csv
```

Expected columns:

```text
text,label,category,source_file,source_id
```

`source_file` and `source_id` are preserved as metadata but are not used as model inputs.

Primary task:

- Binary classification from `text` to `label`.

Optional secondary tasks kept ready in the notebook:

- Multiclass classification from `text` to `category`.

## Model Registry

The notebook will define these presets:

- `wangchanberta`: `airesearch/wangchanberta-base-att-spm-uncased`
- `roberta`: `roberta-base`
- `phayathaibert`: `clicknext/phayathaibert`

`wangchanberta` is the default preset.

## Pipeline Steps

1. Install/import dependencies: `pandas`, `numpy`, `scikit-learn`, `torch`, `datasets`, `transformers`, and `evaluate`.
2. Load the CSV with UTF-8 BOM tolerance.
3. Validate required columns and normalize labels.
4. Print dataset size, label distribution, category distribution, and text length summary.
5. Create stratified train/validation/test splits.
6. Build Hugging Face `DatasetDict` objects.
7. Load tokenizer and sequence-classification model from the selected preset.
8. Tokenize text with truncation and configurable `max_length`.
9. Train with `Trainer`, class-aware metrics, early stopping, and best-model checkpoint loading.
10. Evaluate on the validation and test splits.
11. Run external test predictions for `dataset/test/questions_formatted_id.csv`, normalizing legacy `Id/Instruct/Label/Category` columns when present.
12. Save external test predictions and metrics under the run output directory.
13. Provide an inference helper for quick guardrail predictions.
14. Save the final model, tokenizer, metrics, label mapping, and run configuration under `outputs/finetune/<model>/<task>/<timestamp>/final_model/`.

## Verification

The notebook should be valid nbformat and import-safe. It will not assume the CSV currently exists; instead, it will raise a clear `FileNotFoundError` with the expected path and remediation instructions.

## Notes

- The project `pyproject.toml` currently does not include the ML finetuning stack. The notebook includes an optional install cell so the pipeline can run without immediately changing project dependencies.
- The initial implementation focuses on binary guardrail detection because `label` is the operational allow/block target. `category` is available as the only alternate target.
