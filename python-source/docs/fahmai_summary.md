# FahMai Benchmark Trilogy Finale 2026: Enterprise Data-Agent Showdown

## 1. Overview
**FahMai The Finale** is the culmination of the SuperAI benchmark series. It is an "Enterprise Data-Agent Showdown" where AI agents must answer 150+ business questions by analyzing two fiscal years (2024-2025) of operational data from a multi-channel Thai consumer electronics retailer.

## 2. The Data Universe
The benchmark uses a highly realistic, multimodal dataset (~4.0 GB) encompassing all operations of the fictional "FahMai" brand, which features 5 in-house brands, 10 branches, and over 100K customers. 

The data is split across five distinct surfaces:
* **Tables:** 31 CSVs containing Dimension (who/what/where) and Fact (events) records.
* **Docs (Narrative):** 53,428 Markdown files, mostly in Thai, consisting of policy memos, meeting minutes, and massive volumes of chat logs (LINE OA & LINE WORKS).
* **Renders:** 6,128 image and PDF files simulating scanned documents like bank statements, POS receipts, and vendor invoices. 
* **Logs:** 7,900 operational logs including POS and web tracking data.
* **Reports:** 32 aggregated financial and operational cadence reports.

## 3. Key Technical Challenges
Agents are tested on cross-modal reasoning at scale, facing five compounding difficulties:
1.  **Language Complexity:** A mix of Thai and English, including Buddhist Era (พ.ศ.) dates.
2.  **Multimodality:** Resolving answers by cross-referencing tables, documents, chats, and rendered images.
3.  **Bitemporality:** Factoring in the exact date of an event to apply the correct historical policy.
4.  **Data Quality:** Handling realistic data bugs, typos, and duplicate records.
5.  **Schema Cutovers:** Adapting to structural table changes (e.g., a mid-2025 schema migration).

## 4. Competition Tracks
Competitors must deploy agents locally on provided NTi compute infrastructure, with strict anti-cheat measures (no external API fallbacks like OpenAI or Anthropic).

* **Track 1 (Open-Weight Local):** Any open-weight model hosted on the NTi GPU (1x B200 or LANTA).
* **Track 2 (ThaiLLM-based):** Restricted to models trained primarily in Thai, also hosted locally.
* **OCR Sub-track:** A separate task running on a 32 GB RTX 5090 to extract structured fields (amount, date, account, description) from bank statements into JSON format.

## 5. Evaluation and Scoring
* **Public Leaderboard (Kaggle):** 100 questions scored on exact-token accuracy using a deterministic grader.
* **Private Leaderboard:** 50 held-out questions (Hard, Extremely Hard, and Special tiers) to test generalization.
* **Scoring Formula:** Agents are graded on accuracy and efficiency. The baseline cost is benchmarked at a $94 API-equivalent (using Codex/gpt-5.5-xhigh). Submissions cheaper than the baseline receive a score bonus, while more expensive ones are penalized.

## 6. Final Pitch & Rubric
The final standing combines numeric leaderboard results with a live presentation:
* **50% Leaderboards:** Agentic Private LB (30%) + OCR Private LB (20%).
* **50% Pitch:**
    * **40% Solution:** Evaluating architecture, retrieval harnesses, cost/accuracy trade-offs, adversarial defenses, and data insights.
    * **10% Presentation:** Narrative clarity, reproducibility (1-command run), and live demo quality.
