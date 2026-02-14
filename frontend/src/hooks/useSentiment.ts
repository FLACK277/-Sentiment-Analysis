import { useRef, useState } from "react";
import type { AnalysisResponse } from "../types/sentiment";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8787";

export const useSentiment = () => {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);

  const analyze = async (text: string): Promise<void> => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;

    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/sentiment/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal
      });

      const payload = (await response.json()) as AnalysisResponse | { error?: string };
      if (!response.ok) {
        throw new Error("error" in payload ? payload.error ?? "Failed to analyze sentiment" : "Failed to analyze sentiment");
      }

      setResult(payload as AnalysisResponse);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        return;
      }

      const message = caught instanceof Error ? caught.message : "Unexpected error";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return { result, isLoading, error, analyze };
};
