import type { VercelLikeRequest, VercelLikeResponse } from "../_types";
import { getModelMetadata } from "../../backend/src/services/sentimentService.js";
import { ensureModelReady, sendMethodNotAllowed } from "../_shared";

export default async function handler(req: VercelLikeRequest, res: VercelLikeResponse): Promise<void> {
  if (req.method !== "GET") {
    sendMethodNotAllowed(res, "GET");
    return;
  }

  try {
    await ensureModelReady();
    res.status(200).json(getModelMetadata());
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected model metadata error";
    res.status(500).json({ error: message });
  }
}
