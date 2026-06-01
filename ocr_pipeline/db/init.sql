-- =============================================================
-- db/init.sql — PostgreSQL schema สำหรับ OCR Pipeline
-- รัน auto โดย docker-compose ตอน container สร้างครั้งแรก
-- =============================================================

-- ─── Document type enum ──────────────────────────────────────
CREATE TYPE doc_type_enum AS ENUM (
    'bank_statement',
    'vendor_invoice',
    'warranty_form',
    'receipt',
    'e7_banner',
    't2_doc',
    't3_doc',
    'unknown'
);

-- ─── Job status enum ─────────────────────────────────────────
CREATE TYPE job_status_enum AS ENUM (
    'pending',
    'processing',
    'done',
    'failed'
);

-- ─── Main OCR results table ──────────────────────────────────
CREATE TABLE IF NOT EXISTS ocr_jobs (
    id           BIGSERIAL          PRIMARY KEY,
    artifact_id  VARCHAR(255)       UNIQUE NOT NULL,
    doc_type     doc_type_enum      DEFAULT 'unknown',
    status       job_status_enum    DEFAULT 'pending',
    pred_json    JSONB,                       -- extracted fields
    raw_text     TEXT,                        -- raw OCR text (optional)
    error_msg    TEXT,                        -- error ถ้า failed
    image_b64    TEXT,                        -- เก็บ base64 ถ้าต้องการ
    created_at   TIMESTAMPTZ        DEFAULT NOW(),
    updated_at   TIMESTAMPTZ        DEFAULT NOW()
);

-- ─── Indexes ─────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_artifact  ON ocr_jobs(artifact_id);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status    ON ocr_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_doc_type  ON ocr_jobs(doc_type);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_created   ON ocr_jobs(created_at DESC);

-- full-text search บน pred_json
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_pred_gin  ON ocr_jobs USING GIN (pred_json);

-- ─── Auto-update updated_at trigger ──────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ocr_jobs_updated ON ocr_jobs;
CREATE TRIGGER trg_ocr_jobs_updated
    BEFORE UPDATE ON ocr_jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ─── View: summary ───────────────────────────────────────────
CREATE OR REPLACE VIEW ocr_summary AS
SELECT
    doc_type,
    status,
    COUNT(*)           AS count,
    MIN(created_at)    AS first_at,
    MAX(updated_at)    AS last_at
FROM ocr_jobs
GROUP BY doc_type, status
ORDER BY doc_type, status;
