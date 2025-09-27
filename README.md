# Cybersecurity Impact Extraction Pipeline

An end‑to‑end workflow to **collect cybersecurity news**, **extract full article text**, and **use LLM prompts to classify incidents and quantify economic impact** for a given company and month.

This repository contains:

- `pipeliine.ipynb` — Jupyter notebook that orchestrates the workflow (spelling with double *i* is intentional here to match the uploaded file).
- `utils.py` — Utilities for:
  - Downloading GDELT news for a company in a monthly window
  - Extracting article text from URLs
  - Tagging keyword occurrences
  - Splitting long batches to respect token limits
  - Calling three Gemini/Vertex AI prompts
- `prompt_1.txt` — Extract **cybersecurity‑related monetary values** from passages.
- `prompt_2.txt` — Flag whether each passage **describes a cyberattack related to the named company**.
- `prompt_3.txt` — Consolidate information into **13 dimensions of economic/qualitative impact**.

> Tested with Python 3.9+. Last edited: 2025-09-27 (timezone: Europe/London).

---

## Project structure

```
.
├── pipeliine.ipynb
├── utils.py
├── prompt_1.txt
├── prompt_2.txt
├── prompt_3.txt
└── data/                # created at runtime: raw GDELT CSVs (data/<COMPANY>/gdelt_<Mon_YYYY>.csv)
    └── <COMPANY>/
```

When you run extraction, an `data_extracted/<COMPANY>/` folder is also created with article text and keyword tags.

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install pandas requests tqdm beautifulsoup4 lxml nest_asyncio jupyter \
            google-cloud-aiplatform google-generativeai
```

> `google-cloud-aiplatform` provides the `vertexai` SDK used by `utils.generate`.  
> `google-generativeai` is present as an alternative in the code and can be ignored unless you swap to the non‑Vertex path.

---

## Google Cloud / Vertex AI setup

`utils.generate` is configured to call a Gemini model via **Vertex AI** (default model in code: `gemini-2.5-flash`). To run it:

1. Enable Vertex AI in your GCP project and create credentials.
2. Authenticate locally (either works):
   - **Application Default Credentials**
     ```bash
     gcloud auth application-default login
     ```
   - **Service account JSON**
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
     ```
3. Edit `utils.py` in `generate(...)` and set **your**:
   - `project="YOUR_PROJECT_ID"`
   - optional `location="global"` (or your region)
   - `GenerativeModel("gemini-2.5-flash")` (you can switch to another available Gemini model)

> If you prefer using the `google-generativeai` client instead of Vertex AI, the file already contains a commented alternative you can adapt.

---

## Quick start (Notebook)

1. **Pick a company and month**
   ```python
   company = "Microsoft"
   date    = "May 2021"   # format: 'Mon YYYY'
   ```
2. **Find a safe `days_param` for GDELT** (keeps each request < 250 records):
   ```python
   from utils import try_days_param
   days_param = try_days_param(company, date)
   ```
3. **Download GDELT results to CSV**
   ```python
   from utils import download_csv
   download_csv(company, date, days_param)
   # writes: data/<COMPANY>/gdelt_<Mon_YYYY>.csv
   ```
4. **Fetch and enrich article text**
   ```python
   from utils import download_all
   csv_name   = f"gdelt_{date.replace(' ', '_')}.csv"
   df_articles = download_all(company, csv_name)
   ```
5. **Split into prompt‑sized batches**
   ```python
   from utils import calculate_splits
   splits = calculate_splits(df_articles, prompt="prompt_2.txt", token_limit=50_000)
   ```
6. **Run prompts**
   - **Prompt 2 — attack classification**
     ```python
     from utils import prompt_2_call
     for (start, end) in splits:
         batch_texts = df_articles["text_full"].iloc[start:end].tolist()
         flags = prompt_2_call(batch_texts, company)
         # list of 1/0 for each passage in order
     ```
   - **Prompt 1 — monetary value extraction**
     ```python
     from utils import prompt_1_call
     value_splits = calculate_splits(df_articles, prompt="prompt_1.txt", token_limit=50_000)
     for (start, end) in value_splits:
         batch_texts = df_articles["text_full"].iloc[start:end].tolist()
         results = prompt_1_call(batch_texts)
         # returns: list[{{"value_found": [...], "description_of_value": [...]}}] aligned to inputs
     ```
   - **Prompt 3 — 13‑dimension consolidation**
     > `prompt_3_call` is scaffolded in `utils.py` and may be commented out.  
     > If you plan to use it, open `utils.py` and uncomment/finish the function, or adapt it to your needs.

7. **(Optional) Clean result dictionaries**
   ```python
   from utils import dict_cleaning
   cleaned = dict_cleaning(results_dict)
   ```

---

## How it works (high‑level)

1. **GDELT query window.** For a month like `May 2021`, the code queries from the **15th of the previous month** through the **last day of the following month** and iterates in segments sized by `days_param`. This captures lead/lag coverage around the month and respects GDELT’s per‑query record limit.
2. **Article extraction.** Each GDELT URL is fetched and parsed with `requests` + `BeautifulSoup`; plaintext is stored in `text_full`. The function also tags keyword occurrences (company + a curated cybersecurity keyword list).
3. **LLM prompts.**
   - `prompt_2.txt` returns a **list of 1/0** flags per passage.
   - `prompt_1.txt` returns **monetary values** related to cybersecurity and their descriptions.
   - `prompt_3.txt` consolidates into **13 impact dimensions** (quantitative in USD + qualitative fields).
4. **Token‑aware batching.** `calculate_splits(...)` estimates tokens for the prompt + texts and yields `(start, end)` index windows that fit under a chosen token budget per call.

---

## Configuration notes

- **Paths.**  
  `download_csv(...)` writes to the repo at `data/<COMPANY>/gdelt_<Mon_YYYY>.csv`.  
  `download_all(...)` currently reads from a hard‑coded base path. **Change this line** in `utils.py` to make it repo‑relative:
  ```python
  # utils.py
  uploads_path = "data"  # instead of '/Users/.../tesi/data'
  ```
  It will then write enriched files under `data_extracted/<COMPANY>/`.

- **Model & region.** Tweak in `utils.generate(...)`:
  ```python
  GenerativeModel("gemini-2.5-flash")  # or another available model
  vertexai.init(project="YOUR_PROJECT_ID", location="global")
  ```

- **Token limits.** Examples use `50_000` tokens for prompts 1–2 and `40_000` for prompt 3; adjust per model limits.

---

## API keys & privacy

- Using Vertex AI requires Google Cloud auth as described above. Avoid hard‑coding project IDs and credentials—use environment variables or local config.
- Be mindful of sending third‑party content to an LLM; ensure you have the right to process the text and comply with the model provider’s terms.

---

## Troubleshooting

- **Got 0 results from GDELT:** increase the date window (`days_param`) or try a different month/company spelling.
- **Got >250 results / errors from GDELT:** run `try_days_param(...)` to automatically reduce the window.
- **Long batches / model errors:** lower `token_limit` or use smaller `splits`.
- **Vertex AI auth errors:** verify `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS` and `project` in `utils.generate`.

---

## Acknowledgements

- GDELT Project for news indexing.
- Google Vertex AI / Gemini models for LLM inference.

---

## License

Add a license of your choice (e.g., MIT, Apache-2.0).
