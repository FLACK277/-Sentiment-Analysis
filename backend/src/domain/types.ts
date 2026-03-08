export type SentimentLabel = "positive" | "negative" | "neutral";

export interface ReviewRow {
  product_name: string;
  rating: string;
  title: string;
  review: string;
}

export interface AnalysisRequest {
  text: string;
}

export interface TokenContribution {
  token: string;
  weight: number;
}

export interface ModelScore {
  name: string;
  accuracy: number;
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

export interface DatasetSummary {
  totalRows: number;
  skippedRows: number;
  classDistribution: Record<SentimentLabel, number>;
  sample: Array<{
    product: string;
    title: string;
    sentiment: SentimentLabel;
  }>;
}

export interface ModelMetadata {
  vocabularySize: number;
  trainedRows: number;
  classPrior: Record<"positive" | "negative", number>;
  modelName: string;
  modelSource: string;
  validationAccuracy: number;
  evaluatedModels: ModelScore[];
  topFeatures: TokenContribution[];
}
