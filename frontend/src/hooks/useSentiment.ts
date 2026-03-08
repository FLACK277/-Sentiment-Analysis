import { useRef, useState } from "react";
import type { AnalysisResponse } from "../types/sentiment";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "";

const parseResponse = async <T>(response: Response): Promise<T> => {
  const body = await response.text();
  if (!body.trim()) {
    throw new Error(response.ok ? "The server returned an empty response." : "The request failed without a response body.");
  }

  try {
    return JSON.parse(body) as T;
  } catch {
    throw new Error("The API returned an invalid response. Check that the backend is reachable.");
  }
};

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
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/sentiment/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal
      });

      const payload = await parseResponse<AnalysisResponse | { error?: string }>(response);
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
