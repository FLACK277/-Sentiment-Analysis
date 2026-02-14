import cors from "cors";
import express, { NextFunction, Request, Response } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_PORT } from "./config/constants.js";
import { sentimentRouter } from "./routes/sentimentRoutes.js";
import { loadDatasetSummary } from "./services/datasetService.js";
import { getModelMetadata, trainSentimentModel } from "./services/sentimentService.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "100kb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const datasetPath = path.resolve(__dirname, "../../laptops_dataset_final_600.csv");

let datasetCache: Awaited<ReturnType<typeof loadDatasetSummary>> | null = null;

app.get("/api/health", async (_req: Request, res: Response, next: NextFunction) => {
  try {
    if (!datasetCache) {
      datasetCache = await loadDatasetSummary(datasetPath);
    }

    res.status(200).json({ status: "ok", dataset: datasetCache, model: getModelMetadata() });
  } catch (error) {
    next(error);
  }
});

app.use("/api/sentiment", sentimentRouter);

app.use((_req: Request, res: Response) => {
  res.status(404).json({ error: "Route not found" });
});

app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
  const message = error instanceof Error ? error.message : "Unexpected server error";
  res.status(500).json({ error: message });
});

const port = Number(process.env.PORT) || DEFAULT_PORT;

const bootstrap = async () => {
  const model = await trainSentimentModel(datasetPath);
  console.log(`Model trained with ${model.trainedRows} rows and ${model.vocabularySize} tokens.`);

  app.listen(port, () => {
    console.log(`Sentiment API running on http://localhost:${port}`);
  });
};

bootstrap().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : "Failed to bootstrap server";
  console.error(message);
  process.exit(1);
});
