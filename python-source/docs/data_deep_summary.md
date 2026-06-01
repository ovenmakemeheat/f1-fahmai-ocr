# FahMai Dataset Deep Summary

Source inspected: `data/fah-mai-the-finale-enterprise-data-agentic-showdown/` plus `data/sample_submission.csv`.

This local bundle is the Day-1 analyst view for FahMai, a fictional multi-channel electronics retailer in Thailand. It covers business activity from `2024-01-01` through `2025-12-31`, with the public bundle release dated `2026-01-15`. The corpus is intentionally enterprise-shaped: structured warehouse tables, operational logs, customer/internal conversations, formal documents, rendered document images/PDFs, and cadence reports all describe overlapping parts of the same business.

## 1. Top-Level Inventory

| Surface | Local files | Size | Main purpose |
|---|---:|---:|---|
| `tables/` | 31 CSV | 129.1 MB | Authoritative structured dimensions and facts |
| `docs/` | 53,428 markdown | 129.7 MB | Policies, memos, minutes, emails, customer chats, internal chats, L1 KB |
| `logs/` | 7,935 text logs | 31.7 MB | Daily POS line-item logs, daily web event JSONL, monthly PayWise fee logs |
| `renders/` | 6,128 PNG/PDF | 4.31 GB | Bank statements, receipts, invoices, warranty forms, banners, generated document renders |
| `reports/` | 32 markdown | 45.8 KB | Monthly operations and quarterly finance rollups |
| `README.md` | 1 markdown | 1.9 KB | Bundle data card and caveats |
| `sample_submission.csv` | 1 CSV | 1.5 KB | Kaggle answer template |

The zip file `data/fah-mai-the-finale-enterprise-data-agentic-showdown.zip` is also present and is about 4.25 GB. The summary below describes the extracted folder.

Important local discrepancy: the dataset README says `docs/l1_kb/` has `0` files, but the extracted local tree contains 118 L1 KB markdown files. Treat the extracted local tree as the working corpus, while noting the README count mismatch.

## 2. Competition Answer Surface

`sample_submission.csv` has columns:

| Column | Meaning |
|---|---|
| `id` | Question identifier |
| `response` | Empty answer field to fill |

The visible IDs cover:

- `L3-Q-EASY-001` through `L3-Q-EASY-025`
- `L3-Q-MED-001` through `L3-Q-MED-020`
- `L3-Q-HARD-001` through `L3-Q-HARD-020`
- `L3-Q-XHARD-001` through `L3-Q-XHARD-020`
- Selected reference IDs: `L3-Q-REF-001`, `008`, `019`, `021`, `023`
- Injection/adversarial IDs: `L3-Q-INJ-005`, `009`, `011`, `012`, `013`, `015`, `017`, `018`, `021`, `022`

The submission template itself does not include question text. It only exposes the answer IDs and confirms that injection-style questions are an explicit evaluation category.

## 3. Structured Tables

The table layer is the cleanest starting point for numeric answers. The README states that when structured and narrative sources conflict, database tables are authoritative unless a newer memo or policy version supersedes them.

### 3.1 Dimension Tables

| File | Rows | Role |
|---|---:|---|
| `DIM_BANK_ACCOUNT.csv` | 14 | Bank accounts, banks, roles, branch association, opening balances |
| `DIM_BRANCH.csv` | 11 | HQ, branch, and remote retail locations; service-center and traffic/headcount weights |
| `DIM_CUSTOMER.csv` | 30,000 | B2C/B2B customers, names, contact fields, province/region, account manager, loyalty tier, LINE OA usage |
| `DIM_DATE.csv` | 731 | Full 2024-2025 calendar, Thai public holidays, fiscal year/quarter |
| `DIM_DEPARTMENT.csv` | 9 | Department code, Thai/English name, department type |
| `DIM_EMPLOYEE.csv` | 600 | Employees, branch/department/position, reporting line, status, hire/termination data, canonical leadership tags |
| `DIM_POLICY_VERSION.csv` | 12 | Policy variables, effective windows, value refs, source policy document filenames |
| `DIM_POSITION_LEVEL.csv` | 6 | Position rank and default signing authority |
| `DIM_PRODUCT.csv` | 110 | SKUs, brand family, department, category, price, vendor, lifecycle, warranty, Care Plus eligibility |
| `DIM_PROMO_CAMPAIGN.csv` | 7 | Campaign windows, scope filters, Thai/English descriptions |
| `DIM_VENDOR.csv` | 6 | Vendor identity, category, payment terms, cadence, partner/component flags |
| `DIM_VENDOR_CONTRACT_VERSION.csv` | 22 | Vendor contract versions, effective dates, PDFs, amendment summaries |
| `dim_care_plus_sku_tier.csv` | 2 | Care Plus price/coverage tiers |
| `dim_product_recall_history.csv` | 3 | SKU recall status transitions |
| `dim_promo_mechanic.csv` | 8 | Discount type/value, point multiplier, basket threshold |
| `dim_signing_authority_ladder.csv` | 7 | Approval ceilings, co-signer requirements, department scope |
| `T2_DOC_INVENTORY.csv` | 81 | Rendered document inventory, template/source mapping |

Useful aggregates:

- Branches: 1 HQ, 9 physical branches, 1 remote channel.
- Products: 28 smartphones, 28 laptops, 27 monitors, 27 smartwatches.
- Customers: 29,700 B2C and 300 B2B.
- Loyalty tiers: 7,800 `none`, 7,200 `silver`, 7,500 `gold`, 7,500 `platinum`.

### 3.2 Fact Tables

| File | Rows | Main grain / use |
|---|---:|---|
| `FACT_SALES.csv` | 117,105 | Transaction header: date, branch, customer, employee, channel, totals, shipping, promo, payment, schema version |
| `FACT_SALES_LINE_ITEM.csv` | 309,129 | Transaction line: SKU, quantity, unit price, discount, Care Plus flag, POS log reference |
| `FACT_INVENTORY_MOVEMENT.csv` | 310,827 | Inventory events by SKU/branch/movement type |
| `FACT_INVENTORY_MONTHLY_SNAPSHOT.csv` | 26,220 | Month-end closing units by SKU and branch |
| `FACT_BANK_TRANSACTION.csv` | 65,334 | Bank postings, account, counterparty, related entity/table, amount, balance |
| `FACT_LOYALTY_LEDGER.csv` | 118,857 | Points earn/redeem/adjustment ledger and resulting balance/tier |
| `FACT_PAYROLL.csv` | 14,400 | Pay periods, gross/tax/social/net pay, bank transaction link |
| `FACT_CS_INTERACTION.csv` | 14,368 | Customer-service interaction metadata and links to refunds/warranty/chat sessions |
| `FACT_RETURN.csv` | 7,144 | Product returns, original transaction/line, reason, approver, days since purchase, return amount |
| `FACT_REFUND_PAID.csv` | 7,134 | Refund payment details, approver/co-signer, bank transaction |
| `FACT_WARRANTY_CLAIM.csv` | 3,973 | Warranty claim reason, amount, routing destination, resolution |
| `FACT_SHIPPING.csv` | 23,182 | Shipment tracking, vendor, origin branch, destination province, confirmation status |
| `FACT_VENDOR_PAYMENT.csv` | 809 | Vendor invoices, payment amounts, contract version, approval signers, bank transaction |
| `FACT_PROMO_REDEMPTION.csv` | 1,583 | Promo redemptions by transaction/customer/campaign/channel |

Key fact aggregates verified locally:

- `FACT_SALES` spans `2024-01-01` to `2025-12-31` with total `net_total_thb = 6,514,622,034.20`.
- `FACT_RETURN` spans the same period with `return_amount_thb = 150,367,400.00`.
- `FACT_REFUND_PAID` spans the same period with `refund_amount_thb = 150,299,150.00`.
- `FACT_BANK_TRANSACTION` has total signed `amount_thb = 5,593,708,546.00`.
- `FACT_SALES.schema_version` is date-sensitive:
  - Version `1`: 70,495 rows, `2024-01-01` to `2025-03-31`
  - Version `2`: 46,610 rows, `2025-04-01` to `2025-12-31`
- `FACT_SALES.channel`: 104,449 `offline`, 12,655 `online`, 1 `b2b`.
- `FACT_SALES.payment_method`: credit card, cash, mobile wallet, debit card, bank transfer, and one `net_30` case.

The schema cutover is especially important. Joins and field interpretation should be date-aware around `2025-04-01`; do not assume all sales rows follow one physical/log schema.

## 4. Documents

`docs/` is the largest text surface. It is mostly Thai, sometimes English or mixed Thai/English. Most markdown files are line-oriented JSON-like message transcripts rather than prose markdown.

| Folder | Local files | Size | Content |
|---|---:|---:|---|
| `docs/chat_line_oa/` | 37,441 | 100.5 MB | Customer LINE OA support chats |
| `docs/chat_line_works/` | 15,802 | 27.9 MB | Internal LINE Works threads |
| `docs/l1_kb/` | 118 | 853.7 KB | Customer-support KB: policies, product docs, store info |
| `docs/email/` | 25 | 101.2 KB | All-staff internal emails |
| `docs/minutes/` | 26 | 199.6 KB | Operations meeting minutes |
| `docs/memo/` | 16 | 85.8 KB | Internal policy memos |

### 4.1 Customer LINE OA Chats

These are customer-service transcripts. Each file contains JSON objects with `message_id`, `timestamp`, `speaker`, `message_type`, and `text`. Common topics include:

- Stock checks at specific branches.
- Product recommendations and feature explanations.
- Invoice/payment confirmation, especially B2B cases.
- Order and delivery status.
- Return/refund/warranty support.
- Polite Thai service language with customer names and CS-agent identities.

These chats are useful for questions that require exact customer interaction evidence, but they are not the safest first source for numeric totals. Link them through `FACT_CS_INTERACTION.chat_session_id` or other identifiers when possible.

### 4.2 Internal LINE Works Chats

These are internal operational threads among employees, often with action items and handoffs. The samples include:

- CS-to-operations case follow-up.
- Delivery route/status checks.
- Executive or leadership transition claims embedded in `[[CLAIM:...]]` markers.
- Owner/status/next-action summaries.

LINE Works is likely important for hard questions involving "who knew what when", leadership transition details, approval flows, handover dates, or operational exceptions. Treat claims as evidence, but reconcile them against tables, memos, and policy versions.

### 4.3 L1 Knowledge Base

The local L1 KB has:

- `policies/`: 5 files, including cancellation, membership points, return, shipping, and warranty policies.
- `products/`: 110 files, matching the product catalog scale.
- `store_info/`: 3 files, including general/store-facing information.

The KB appears updated beyond the 2024-2025 data window in at least some files, e.g. policy pages with `วันที่อัปเดต: 1 มีนาคม 2569` (`2026-03-01`). For historical questions, use effective-date logic and compare against `DIM_POLICY_VERSION` and memos.

Notable security-adjacent KB mentions:

- `store_info/general_faq.md` states FahMai does not accept cryptocurrency payments.
- `policies/warranty_policy.md` excludes or conditions issues caused by third-party software, device rooting, or device jailbreak.

### 4.4 Memos, Minutes, Emails

These formal narrative documents describe internal changes, operations discipline, and policy updates.

- Memos include policy changes such as return-window updates, signing authority, approval routing, and operational standardization.
- Meeting minutes are monthly operations records with attendees such as operations and CS leads. They discuss handoff quality, issue logs, owner tracking, and follow-up process.
- Emails are all-staff communications about leadership transition, business continuity, workflow, escalation, and official communication channels.

These documents can supersede earlier table/policy assumptions when they are newer and explicitly effective. They are also useful for interpreting conflicts between operational practice and structured data.

## 5. Logs

`logs/` contains 7,935 files:

| Log group | Files | Size | Format | Grain |
|---|---:|---:|---|---|
| POS daily logs | 7,196 | 17.7 MB | TSV | Branch/day line-item events |
| Web daily logs | 731 | 14.0 MB | JSONL | Web session events |
| PayWise fee logs | 8 | 728 bytes | CSV | Monthly fee files |

Sample POS fields:

`timestamp`, `branch_code`, `txn_id`, `line_seq`, `sku_id`, `quantity`, `unit_price_thb`, `discount_amt`, `payment_method`, `schema_version`

Sample web event fields:

`timestamp`, `session_id`, `customer_id`, `event_type`, `sku_id`, `referrer`, `user_agent_hash`, `ip_region`, `txn_id`

The logs are reconciliation surfaces. They help validate or debug fact-table rows, especially:

- POS line item consistency against `FACT_SALES_LINE_ITEM`.
- Web funnel behavior before online transactions.
- Payment-fee details where PayWise is involved.
- Schema cutover behavior, because POS logs also carry `schema_version`.

## 6. Renders

`renders/` is the dominant disk footprint and contains scanned/generated business documents.

| Render type | Files | Size | Format |
|---|---:|---:|---|
| `bank_statement/` | 2,714 | 1.28 GB | PNG |
| `warranty_form/` | 1,963 | 1.99 GB | PNG |
| `vendor_invoice/` | 792 | 743.2 MB | PNG |
| `receipt/` | 563 | 280.1 MB | PNG |
| `t2_doc/` | 81 | 1.5 MB | PDF |
| `t3_doc/` | 11 | 20.1 MB | PNG |
| `e7_banner/` | 4 | 1.7 MB | PNG |

These files are for OCR/multimodal grounding. The tables reference render filenames through standard IDs or source inventory rows where applicable. Expected uses:

- Bank statement OCR and reconciliation against `FACT_BANK_TRANSACTION`.
- Vendor invoice verification against `FACT_VENDOR_PAYMENT` and contract versions.
- Receipt validation against sales/line items.
- Warranty-form extraction against `FACT_WARRANTY_CLAIM`.
- T2/T3 generated documents for document-inventory questions.
- Promo/banner visual checks for campaign questions.

## 7. Reports

`reports/` has 32 markdown files:

- 24 monthly OPS reports: `OPS_REPORT_YYYY-MM.md` for every month in 2024 and 2025.
- 8 quarterly FIN close reports: `FIN_CLOSE_YYYY-QN.md` for every quarter in 2024 and 2025.

OPS reports contain:

- Monthly revenue summary.
- B2C/B2B split and transaction counts.
- Online/offline counts.
- Top SKUs by revenue.
- Per-branch performance.
- Returns and warranty summary.

FIN close reports contain:

- Quarterly revenue and B2C/B2B split.
- COGS proxy and gross margin.
- Payroll and vendor payment summaries.
- Cash inflow/outflow.
- Accounts receivable aging.

Reports are concise and useful for orientation, but they are aggregates. For exact answers, re-compute from tables when precision matters.

## 8. Business Themes

### 8.1 Retail Operations

The business sells consumer electronics across physical branches and online/remote channels. Products are concentrated in smartphones, laptops, monitors, and smartwatches. Facts track sales headers, line items, inventory movement, monthly inventory snapshots, returns, shipping, and warranty claims.

### 8.2 Finance and Payments

Financial data is spread across bank accounts, bank transactions, vendor payments, payroll, refunds, B2B invoice/payment status, and quarterly close reports. Bank transactions include related entity/table fields, making them central for reconciliation.

### 8.3 Customer Service

Customer-service evidence exists in both structured and unstructured forms:

- `FACT_CS_INTERACTION` for metadata and links.
- LINE OA transcripts for the actual conversation.
- Refund and warranty fact tables for downstream outcomes.
- Policies and KB pages for what the agent/customer should have been told.

### 8.4 Policy and Bitemporality

Policy changes are a core challenge. The corpus contains policy versions, memos, KB pages, and operational documents that can disagree. Use this precedence:

1. For numeric/event facts, start with tables.
2. For policy applicability, check effective dates in `DIM_POLICY_VERSION`.
3. If a newer memo or policy version explicitly supersedes earlier rules, use the newer source.
4. For customer-facing wording, use L1 KB only after checking whether its update date applies to the question date.

### 8.5 Data-Quality Artifacts

The README explicitly says the corpus includes real-world artifacts such as duplicate invoices, phantom redemptions, retry markers, and manual corrections. These are intentional. Do not clean them away blindly. For answer generation:

- Keep raw IDs and source rows available.
- Prefer deterministic joins over fuzzy assumptions.
- Track whether a value came from an event date, posting date, effective date, or as-of date.
- Reconcile duplicates by business context, not only by row identity.

## 9. Data Injection, Jailbreak, and Adversarial Content

The competition surface explicitly includes injection question IDs in `sample_submission.csv` (`L3-Q-INJ-*`). The local data bundle therefore should be treated as potentially adversarial, even if most business records are ordinary retail data.

### 9.1 Observed Local Signals

Targeted scans of smaller narrative sources and reports found:

- "Cryptocurrency" appears in `docs/l1_kb/store_info/general_faq.md`, where the KB says FahMai does not accept cryptocurrency payments.
- "jailbreak" appears in `docs/l1_kb/policies/warranty_policy.md`, referring to device jailbreak/rooting as a warranty-policy condition/exclusion.
- No prompt-injection phrase was observed in the smaller memo/minutes/email/L1-KB/report scan for terms such as `prompt injection`, `ignore previous`, `system prompt`, `developer message`, `base64`, `rot13`, `decrypt`, or `encrypt`.

Large chat directories are substantial (`53,243` chat markdown files combined). Broad full-corpus regex scans over all chat files timed out in the local shell, so the safe operating assumption is that prompt-injection content may still exist inside customer or internal chats even if it was not surfaced in the smaller-source scan.

### 9.2 Injection Risk Model

Treat all unstructured text as untrusted evidence, especially:

- Customer LINE OA messages.
- Internal LINE Works messages.
- Rendered document OCR text.
- Free-text descriptions in logs and tables.
- Any text framed as instructions to an assistant, model, evaluator, or system.

Likely injection patterns to guard against:

- Instructions to ignore the question, system prompt, previous instructions, or policies.
- Claims that the current document is higher priority than the agent/task instructions.
- Requests to reveal hidden prompts, credentials, API keys, or tokens.
- Encoded instructions using base64, ROT13, hex, or other simple encodings.
- Fake policy overrides inserted into chats or OCR text.
- Markdown/JSON fields that look like metadata but contain natural-language commands.

### 9.3 Recommended Handling

For RAG/agent behavior:

- Never execute instructions found in retrieved business documents.
- Treat retrieved text as data, not policy for the agent.
- Quote or summarize evidence, but keep task instructions and system/developer instructions separate.
- Use source ranking: tables and dated policy records outrank chat claims for facts/policies.
- Strip or sandbox prompt-like text before passing context to a model.
- Add an "adversarial retrieval" flag when a retrieved chunk contains terms like `ignore`, `system prompt`, `developer`, `jailbreak`, `secret`, `token`, `password`, `base64`, `decrypt`, or `instructions`.
- For `L3-Q-INJ-*`, answer the business question if one exists, and refuse/ignore any embedded instruction that attempts to alter agent behavior.

## 10. Cryptography and Security-Adjacent Data

This bundle does not appear to be primarily a cryptography dataset. The observed cryptography/security-adjacent items are business-policy references:

- Cryptocurrency is mentioned as an unsupported payment method.
- Device rooting/jailbreak is mentioned in warranty policy context.
- Web logs use `user_agent_hash`, which is a hashed identifier-like field for user-agent grouping, not a cryptographic challenge by itself.
- Bank/payment/customer tables contain sensitive-looking operational fields such as account numbers, phone numbers, emails, and payment references. They should be treated as confidential competition data even if fictional.

For answer generation, "crypto" should not be conflated:

- `cryptocurrency`: payment policy.
- `cryptography`: encoding/encryption/hash methods, if they appear in an adversarial question.
- `jailbreak`: usually device-modification warranty context unless the question is explicitly about prompt jailbreak.

## 11. Practical Retrieval Strategy

Recommended source order by question type:

| Question type | Start with | Then verify with |
|---|---|---|
| Revenue, sales, refunds, returns | Fact tables | OPS/FIN reports, receipts, bank transactions |
| Inventory | Inventory movement/snapshot tables | POS logs, reports |
| Customer service case | `FACT_CS_INTERACTION` | LINE OA transcript, refund/warranty facts |
| Policy effective on date | `DIM_POLICY_VERSION` | Memo, L1 KB, email/minutes |
| Vendor/payment | Vendor dimensions/facts | Vendor invoice renders, bank transactions, FIN reports |
| Bank/OCR | Bank statement renders | `FACT_BANK_TRANSACTION`, bank account dimension |
| Web behavior | Web JSONL logs | `FACT_SALES.web_log_line_id`, sales table |
| Internal approval/leadership | Employee/policy tables | LINE Works, memos, all-staff emails |
| Injection/adversarial | Submission ID plus retrieved docs | Apply untrusted-text handling before answering |

## 12. Key Caveats

- Most narrative is Thai; Thai tokenization and Buddhist Era dates matter.
- Some local documents have 2026 update dates even though the fiscal event window is 2024-2025.
- The sales schema changes on `2025-04-01`.
- Reports are summaries, not authoritative replacements for fact-table computation.
- Rendered files require OCR/vision extraction before text-only analysis.
- The dataset intentionally includes inconsistent or messy business artifacts.
- The local README count for `l1_kb` does not match the extracted files.

