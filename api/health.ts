import type { VercelRequest, VercelResponse } from "@vercel/node";
import { getModelMetadata } from "../backend/src/services/sentimentService";
import { ensureModelReady, getDatasetSummaryCached, sendMethodNotAllowed } from "./_shared";

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  if (req.method !== "GET") {
    sendMethodNotAllowed(res, "GET");
    return;
  }

  try {
    await ensureModelReady();
    const dataset = await getDatasetSummaryCached();
    const model = getModelMetadata();

    res.status(200).json({ status: "ok", dataset, model });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected health error";
    res.status(500).json({ error: message });
  }
}
