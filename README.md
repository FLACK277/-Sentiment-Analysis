# 3D Sentiment Analysis Platform (TypeScript Full-Stack)

A production-style, TypeScript-first refactor of the original notebook workflow.

## Stack
- **Backend**: Node.js + Express + TypeScript
- **Frontend**: React + Vite + TypeScript + React Three Fiber
- **Modeling**: In-repo Naive Bayes text classifier trained from `laptops_dataset_final_600.csv` at server startup

## What changed (review + refactor)

### 1) Correctness and architecture
- Replaced notebook-only flow with modular backend services, typed domain contracts, and route isolation.
- Added startup bootstrap that trains the model once, then serves analysis requests.
- Added centralized server error middleware and safer async handling in `/api/health`.

### 2) ML/NLP improvements
- Replaced generic lexicon scoring with a dataset-adapted **multinomial Naive Bayes** pipeline.
- Uses deterministic preprocessing + tokenization before both training and inference.
- Adds `topContributors` token weights in API output for explainability.
- Returns neutral fallback when input is low-signal after preprocessing.

### 3) Frontend improvements
- Added cancellation-safe request hook (`AbortController`) to prevent stale-response UI bugs.
- Added confidence bar and token contribution list.
- Maintained interactive 3D card with sentiment color coding and smooth motion.

## Project structure

```text
.
├── backend
│   └── src
│       ├── config
│       ├── domain
│       ├── routes
│       ├── services
│       ├── utils
│       └── server.ts
├── frontend
│   └── src
│       ├── components
│       ├── hooks
│       ├── types
│       ├── App.tsx
│       └── main.tsx
└── laptops_dataset_final_600.csv
```

## Run locally

### 1) Install dependencies
```bash
npm install
```

### 2) Run services
```bash
npm run dev -w backend
```
```bash
npm run dev -w frontend
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8787`

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
  "explanation": "Naive Bayes margin 0.891 with 5 processed tokens.",
  "topContributors": [
    { "token": "excellent", "weight": 0.71 },
    { "token": "smooth", "weight": 0.54 }
  ]
}
```

### GET `/api/sentiment/model-info`
Returns training metadata (`vocabularySize`, `trainedRows`, class priors).

### GET `/api/health`
Returns server status + dataset summary + model metadata.

## 3D visualization
- 3D rounded card rotates/floats and updates by sentiment state.
- Color mapping: positive (green), neutral (blue), negative (red).
- Side panel shows confidence, tokens, and top weighted token contributions.
