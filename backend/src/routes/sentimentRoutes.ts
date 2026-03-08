import { Request, Response, Router } from "express";
import { analyzeText, getModelMetadata } from "../services/sentimentService.js";
import type { AnalysisRequest } from "../domain/types.js";

export const sentimentRouter = Router();

sentimentRouter.get("/model-info", (_req: Request, res: Response) => {
  res.status(200).json(getModelMetadata());
});

sentimentRouter.post("/analyze", async (req: Request<unknown, unknown, AnalysisRequest>, res: Response) => {
  try {
    const text = typeof req.body?.text === "string" ? req.body.text : "";
    const result = await analyzeText(text);
    res.status(200).json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected analysis error";
    res.status(400).json({ error: message });
  }
});
