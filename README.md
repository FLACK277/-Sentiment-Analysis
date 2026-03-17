# Sentiment Analysis Platform 

A Vite + Tailwind frontend with a local Node API that delegates inference to a Python model derived from LaptopML.ipynb.

## Stack
- **Backend (local/dev)**: Node.js + Express + TypeScript
- **Model runtime**: Python + scikit-learn + NLTK
- **Frontend**: React + Vite + TypeScript + Tailwind CSS
- **Dataset**: `laptops_dataset_final_600.csv`

##

## Project structure

```text
.
├── api/                          # Vercel serverless API (Option B)
│   ├── _shared.ts
│   ├── health.ts
│   └── sentiment/
│       ├── analyze.ts
│       └── model-info.ts
├── backend/                      # Local Node server runtime
├── frontend/
├── vercel.json
└── laptops_dataset_final_600.csv
```

## Run locally

### 1) Create the Python environment
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2) Install Node dependencies
```bash
npm install
```

### 3) Run services
```bash
npm run dev
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8787`

The first backend start may take longer because it trains and caches a notebook-backed model artifact under `backend/ml/artifacts/`.

### 3) Build
```bash
npm run build
```

## API

### POST `/api/sentiment/analyze`
```json
{ "text": "Battery life is excellent and performance is smooth" }
```

Example response:
```json
{
  "text": "Battery life is excellent and performance is smooth",
  "sentiment": "positive",
  "score": 0.891,
  "confidence": 0.446,
  "tokens": ["battery", "life", "excellent", "performance", "smooth"],
  "explanation": "Random Forest from LaptopML.ipynb predicted positive using 5 processed tokens.",
  "topContributors": [
    { "token": "excellent", "weight": 0.71 },
    { "token": "smooth", "weight": 0.54 }
  ]
}
```

### GET `/api/sentiment/model-info`
Returns training metadata including the selected notebook model, validation accuracy, vocabulary size, and top features.

### GET `/api/health`
Returns server status + dataset summary + model metadata.

## Deploy to Vercel (Full Stack)

This repository now supports full-stack deployment on Vercel using:
- static frontend build from `frontend/`
- Python serverless functions from `api/**/*.py`

### 1) Push your repository
Push the latest code to GitHub/GitLab/Bitbucket.

### 2) Import project in Vercel
1. Open Vercel dashboard.
2. Create a new project and import this repository.
3. Keep root directory as `.`.

### 3) Build settings
These are already defined in `vercel.json`:
- `installCommand`: `npm install`
- `buildCommand`: `npm run build -w frontend`
- `outputDirectory`: `frontend/dist`

### 4) Python dependencies
Vercel installs Python dependencies from root `requirements.txt` for Python functions.

### 5) Environment variables
For same-origin deployment, no API base URL is required.
Do not set `VITE_API_URL` unless you intentionally want to call a different backend domain.

### 6) Deploy and verify
After deployment, test:
- `/` for UI
- `/api/health`
- `/api/sentiment/model-info`
- `/api/sentiment/analyze` with POST JSON body: `{ "text": "great battery life" }`

Or run the smoke test script:
```bash
npm run smoke:vercel -- https://your-app.vercel.app
```

## Notes

- Local backend mode still works with Node + Python (`npm run dev`).
- On Vercel, API routes are served by Python handlers in `api/` and are independent of local `.venv` paths.
