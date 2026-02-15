import type { VercelRequest, VercelResponse } from "@vercel/node";
import { analyzeText } from "../../backend/src/services/sentimentService";
import { ensureModelReady, sendMethodNotAllowed } from "../_shared";

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  if (req.method !== "POST") {
    sendMethodNotAllowed(res, "POST");
    return;
  }

  try {
    await ensureModelReady();

    const payload = req.body as { text?: unknown } | undefined;
    const text = typeof payload?.text === "string" ? payload.text : "";
    const result = analyzeText(text);

    res.status(200).json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected analysis error";
    res.status(400).json({ error: message });
  }
}
