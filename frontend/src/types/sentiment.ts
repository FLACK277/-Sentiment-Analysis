export type SentimentLabel = "positive" | "negative" | "neutral";

export interface TokenContribution {
  token: string;
  weight: number;
}

export interface AnalysisResponse {
  text: string;
  sentiment: SentimentLabel;
  score: number;
  confidence: number;
  tokens: string[];
  explanation: string;
  topContributors: TokenContribution[];
}
