import type { VercelResponse } from "@vercel/node";
import path from "node:path";
import { loadDatasetSummary } from "../backend/src/services/datasetService";
import { trainSentimentModel } from "../backend/src/services/sentimentService";

const datasetPath = path.resolve(process.cwd(), "laptops_dataset_final_600.csv");

let modelInitPromise: Promise<void> | null = null;
let datasetSummaryPromise: ReturnType<typeof loadDatasetSummary> | null = null;

export const ensureModelReady = async (): Promise<void> => {
  if (!modelInitPromise) {
    modelInitPromise = trainSentimentModel(datasetPath).then(() => undefined);
  }

  await modelInitPromise;
};

export const getDatasetSummaryCached = async () => {
  if (!datasetSummaryPromise) {
    datasetSummaryPromise = loadDatasetSummary(datasetPath);
  }

  return datasetSummaryPromise;
};

export const sendMethodNotAllowed = (res: VercelResponse, allowed: string): void => {
  res.setHeader("Allow", allowed);
  res.status(405).json({ error: "Method not allowed" });
};
