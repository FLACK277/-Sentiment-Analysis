# 3D Sentiment Analysis Platform 

A production-style, TypeScript-first refactor of the original notebook workflow.

## Stack
- **Backend (local/dev)**: Node.js + Express + TypeScript
- **Backend (Vercel prod)**: Serverless functions under `api/`
- **Frontend**: React + Vite + TypeScript + React Three Fiber
- **Modeling**: In-repo Naive Bayes text classifier trained from `laptops_dataset_final_600.csv`

## What changed (review + refactor)

### 1) Correctness and architecture
- Replaced notebook-only flow with modular backend services, typed domain contracts, and route isolation.
- Added startup bootstrap that trains the model once for local Node server mode.
- Added serverless initialization guards for Vercel (`api/_shared.ts`) so model training occurs lazily and safely per function instance.

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

## Deploy to Vercel

This repository is optimized for seamless deployment on Vercel, with both the frontend and serverless API functions hosted together on a single project.

### Prerequisites

- A [Vercel account](https://vercel.com/signup) (free tier works)
- Your repository on GitHub, GitLab, or Bitbucket
- Node.js 18+ (handled automatically by Vercel)

### Deployment Steps

#### 1. Prepare Your Repository

Ensure your repository includes:
- `vercel.json` configuration file (already included)
- `package.json` with `@vercel/node` dependency
- Root-level `tsconfig.json` for API compilation
- API functions in the `api/` directory

All of these are already configured in this repository.

#### 2. Connect to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..."** → **"Project"**
3. Import your Git repository
4. Select the repository: `FLACK277/-Sentiment-Analysis`

#### 3. Configure Project Settings

Vercel will auto-detect the configuration from `vercel.json`:

- **Framework Preset**: Other (or Vite if detected)
- **Root Directory**: `.` (keep as repository root - do NOT change)
- **Build Command**: `npm run build -w frontend` (auto-configured)
- **Output Directory**: `frontend/dist` (auto-configured)
- **Install Command**: `npm install` (auto-configured)

**Important**: Do not change the root directory. The API functions in the `api/` folder must be at the repository root for Vercel to detect them as serverless functions.

#### 4. Deploy

Click **"Deploy"** and wait for the build to complete (typically 1-3 minutes).

#### 5. Verify Deployment

After deployment, test your endpoints:

**Frontend:**
- `https://your-project.vercel.app/` - Main application UI

**API Endpoints:**
- `https://your-project.vercel.app/api/health` - Health check
- `https://your-project.vercel.app/api/sentiment/analyze` - POST sentiment analysis
- `https://your-project.vercel.app/api/sentiment/model-info` - GET model metadata

### How It Works

#### Serverless API Functions

The `api/` directory contains serverless functions that:
- Use official `@vercel/node` types (`VercelRequest`, `VercelResponse`)
- Lazy-load and cache the sentiment model on first request
- Share model state across invocations within the same instance
- Import backend services directly (TypeScript files, not compiled JS)

Each API route is a separate serverless function:
```
api/
├── _shared.ts         # Shared utilities (model caching, helpers)
├── health.ts          # GET /api/health
└── sentiment/
    ├── analyze.ts     # POST /api/sentiment/analyze
    └── model-info.ts  # GET /api/sentiment/model-info
```

#### Configuration Files

**`vercel.json`:**
```json
{
  "buildCommand": "npm run build -w frontend",
  "outputDirectory": "frontend/dist",
  "rewrites": [
    { "source": "/((?!api(/|$)).*)", "destination": "/index.html" }
  ]
}
```

The `rewrites` rule ensures:
- API routes (`/api/*`) are handled by serverless functions
- All other routes fall back to `index.html` for SPA routing

**Root `tsconfig.json`:**
Provides TypeScript configuration for API functions with proper module resolution.

### Environment Variables

**Default (Same-Origin) Deployment:**
No environment variables required. The frontend automatically uses relative URLs (`/api/...`) which resolve to the same domain.

**Custom Backend (Optional):**
If you need the frontend to connect to a different API backend:
1. Go to Project Settings → Environment Variables
2. Add: `VITE_API_URL=https://your-api-domain.com`
3. Redeploy

### Troubleshooting

**Build Fails:**
- Ensure `@vercel/node` is in root `package.json` devDependencies
- Check that `tsconfig.json` exists at repository root
- Verify Node.js version compatibility (18+ recommended)

**API Functions Return 404:**
- Confirm `api/` directory is at repository root (not in a subdirectory)
- Check that functions export a default handler: `export default async function handler(req, res) { ... }`

**Frontend Can't Connect to API:**
- Verify no `VITE_API_URL` environment variable conflicts
- Check browser console for CORS or network errors
- Test API endpoints directly: `curl https://your-app.vercel.app/api/health`

**Model Training Timeout:**
- Vercel functions have a 10-second timeout on the free tier
- Model training (~600 rows) typically completes in 1-2 seconds
- For larger datasets, consider pre-training and loading a serialized model

### Continuous Deployment

Vercel automatically redeploys when you push to your repository's main branch:
1. Push changes to GitHub: `git push origin main`
2. Vercel detects the push and triggers a new deployment
3. Your site updates automatically (usually within 2-3 minutes)

### Production Considerations

- **Cold Starts**: First request after inactivity may take 2-5 seconds while the function initializes
- **Model Caching**: The trained model persists in memory during function lifetime (typically 5-15 minutes of inactivity)
- **Concurrency**: Vercel automatically scales serverless functions based on traffic
- **Logs**: View function logs in Vercel Dashboard → Project → Functions tab

## 3D visualization
- 3D rounded card rotates/floats and updates by sentiment state.
- Color mapping: positive (green), neutral (blue), negative (red).
- Side panel shows confidence, tokens, and top weighted token contributions.
