# KrishiMitra AI — Real-Data-First Upgrade

This upgrade removes synthetic weather, mandi prices, and government-scheme results from the runnable MVP.

## Verified source strategy

- **Weather:** India Meteorological Department (IMD) / Meghdoot APIs.
- **Market:** Government of India Open Government Data (OGD) / AGMARKNET resource.
- **Government schemes:** Government of India OGD resource, configured by resource ID.

The app returns an explicit unavailable/error state if a live source or credential is missing. It does **not** substitute made-up agricultural values.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env  # Linux/macOS
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Government OGD setup

Create/configure an OGD API key and put the current resource IDs for the mandi and scheme datasets in `.env`. Do not hard-code credentials.

## Important

The existing legacy Python modules in this archive are preserved for reference. The new `app.py` + `services/real_data.py` path is the clean real-data-first MVP entry point.
