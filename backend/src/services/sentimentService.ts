import { readFile } from "node:fs/promises";
import { parse } from "csv-parse/sync";
import {
  AnalysisResponse,
  ModelMetadata,
  ReviewRow,
  SentimentLabel,
  TokenContribution
} from "../domain/types.js";
import { tokenize } from "../utils/text.js";
import { POSITIVE_RATING_THRESHOLD } from "../config/constants.js";

interface WordStats {
  positive: number;
  negative: number;
}

interface ModelState {
  priors: { positive: number; negative: number };
  tokenStats: Map<string, WordStats>;
  tokenTotals: { positive: number; negative: number };
  vocabularySize: number;
  trainedRows: number;
}

let modelState: ModelState | null = null;

const labelFromRating = (ratingRaw: string): "positive" | "negative" | null => {
  const rating = Number(ratingRaw);
  if (Number.isNaN(rating)) return null;
  return rating >= POSITIVE_RATING_THRESHOLD ? "positive" : "negative";
};

const smoothLogProb = (count: number, total: number, vocabularySize: number): number =>
  Math.log((count + 1) / (total + vocabularySize));

const toFinalLabel = (score: number): SentimentLabel => {
  if (score > 0.45) return "positive";
  if (score < -0.45) return "negative";
  return "neutral";
};

const toConfidence = (raw: number): number => {
  const bounded = Math.min(1, Math.max(0, Math.abs(raw) / 2));
  return Number(bounded.toFixed(3));
};

const toContributors = (
  tokens: string[],
  tokenStats: Map<string, WordStats>,
  totals: { positive: number; negative: number },
  vocabularySize: number
): TokenContribution[] => {
  const tokenWeights = tokens.map((token) => {
    const stats = tokenStats.get(token) ?? { positive: 0, negative: 0 };
    const positiveWeight = smoothLogProb(stats.positive, totals.positive, vocabularySize);
    const negativeWeight = smoothLogProb(stats.negative, totals.negative, vocabularySize);
    return { token, weight: Number((positiveWeight - negativeWeight).toFixed(3)) };
  });

  return tokenWeights
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 6);
};

export const trainSentimentModel = async (datasetPath: string): Promise<ModelMetadata> => {
  const csvText = await readFile(datasetPath, "utf8");
  const rows = parse(csvText, { columns: true, skip_empty_lines: true, trim: true }) as ReviewRow[];

  const tokenStats = new Map<string, WordStats>();
  const tokenTotals = { positive: 0, negative: 0 };
  let positiveRows = 0;
  let negativeRows = 0;
  let trainedRows = 0;

  for (const row of rows) {
    const label = labelFromRating(row.rating);
    if (!label || !row.review?.trim()) {
      continue;
    }

    const merged = `${row.title ?? ""} ${row.review ?? ""}`;
    const tokens = tokenize(merged);
    if (tokens.length === 0) {
      continue;
    }

    trainedRows += 1;
    if (label === "positive") positiveRows += 1;
    if (label === "negative") negativeRows += 1;

    for (const token of tokens) {
      const stats = tokenStats.get(token) ?? { positive: 0, negative: 0 };
      stats[label] += 1;
      tokenStats.set(token, stats);
      tokenTotals[label] += 1;
    }
  }

  const totalRows = Math.max(positiveRows + negativeRows, 1);
  modelState = {
    priors: {
      positive: positiveRows / totalRows,
      negative: negativeRows / totalRows
    },
    tokenStats,
    tokenTotals,
    vocabularySize: Math.max(tokenStats.size, 1),
    trainedRows
  };

  return {
    vocabularySize: modelState.vocabularySize,
    trainedRows,
    classPrior: {
      positive: Number(modelState.priors.positive.toFixed(4)),
      negative: Number(modelState.priors.negative.toFixed(4))
    }
  };
};

export const getModelMetadata = (): ModelMetadata => {
  if (!modelState) {
    throw new Error("Model is not initialized yet.");
  }

  return {
    vocabularySize: modelState.vocabularySize,
    trainedRows: modelState.trainedRows,
    classPrior: {
      positive: Number(modelState.priors.positive.toFixed(4)),
      negative: Number(modelState.priors.negative.toFixed(4))
    }
  };
};

export const analyzeText = (input: string): AnalysisResponse => {
  if (!modelState) {
    throw new Error("Model not initialized. Call trainSentimentModel before analyzeText.");
  }

  const text = input?.trim();
  if (!text) {
    throw new Error("Input text is required.");
  }

  const tokens = tokenize(text);
  if (tokens.length === 0) {
    return {
      text,
      sentiment: "neutral",
      score: 0,
      confidence: 0,
      tokens: [],
      explanation: "No meaningful tokens remained after preprocessing.",
      topContributors: []
    };
  }

  const { priors, tokenStats, tokenTotals, vocabularySize } = modelState;
  let positiveLog = Math.log(Math.max(priors.positive, Number.EPSILON));
  let negativeLog = Math.log(Math.max(priors.negative, Number.EPSILON));

  for (const token of tokens) {
    const stats = tokenStats.get(token) ?? { positive: 0, negative: 0 };
    positiveLog += smoothLogProb(stats.positive, tokenTotals.positive, vocabularySize);
    negativeLog += smoothLogProb(stats.negative, tokenTotals.negative, vocabularySize);
  }

  const margin = positiveLog - negativeLog;
  const normalized = margin / Math.max(tokens.length, 1);
  const sentiment = toFinalLabel(normalized);
  const confidence = toConfidence(normalized);
  const topContributors = toContributors(tokens, tokenStats, tokenTotals, vocabularySize);

  return {
    text,
    sentiment,
    score: Number(normalized.toFixed(3)),
    confidence,
    tokens,
    explanation: `Naive Bayes margin ${normalized.toFixed(3)} with ${tokens.length} processed tokens.`,
    topContributors
  };
};
