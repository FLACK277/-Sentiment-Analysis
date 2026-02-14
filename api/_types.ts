export interface VercelLikeRequest {
  method?: string;
  body?: unknown;
}

export interface VercelLikeResponse {
  setHeader(name: string, value: string): void;
  status(code: number): { json(payload: unknown): void };
}
