# Sveta Colours — Image Pipeline

Local pipeline that downloads artwork photos from Google Drive, processes them (crop, correct, normalize), and re-uploads the results.

## Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline script |
| `config.example.json` | Template for local config |
| `config.json` | Your local config (git-ignored) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
cd tools/image_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
# Edit config.json — fill in your Google Drive folder IDs
```

## Run

```bash
python run_pipeline.py
```

By default `dry_run` is `true` — no files will be uploaded. Set it to `false` in `config.json` when ready.
