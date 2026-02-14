import { createReadStream } from "node:fs";
import { parse } from "csv-parse";
import { POSITIVE_RATING_THRESHOLD, PREVIEW_SAMPLE_SIZE } from "../config/constants.js";
import { DatasetSummary, ReviewRow, SentimentLabel } from "../domain/types.js";

const toLabelFromRating = (ratingRaw: string): SentimentLabel => {
  const rating = Number(ratingRaw);
  if (Number.isNaN(rating)) return "neutral";
  return rating >= POSITIVE_RATING_THRESHOLD ? "positive" : "negative";
};

export const loadDatasetSummary = async (datasetPath: string): Promise<DatasetSummary> => {
  const classDistribution: Record<SentimentLabel, number> = {
    positive: 0,
    negative: 0,
    neutral: 0
  };

  const sample: DatasetSummary["sample"] = [];
  let totalRows = 0;
  let skippedRows = 0;

  return new Promise((resolve, reject) => {
    createReadStream(datasetPath)
      .pipe(parse({ columns: true, skip_empty_lines: true, trim: true }))
      .on("data", (row: ReviewRow) => {
        totalRows += 1;
        if (!row.review?.trim()) {
          skippedRows += 1;
          return;
        }

        const sentiment = toLabelFromRating(row.rating);
        classDistribution[sentiment] += 1;

        if (sample.length < PREVIEW_SAMPLE_SIZE) {
          sample.push({
            product: row.product_name,
            title: row.title,
            sentiment
          });
        }
      })
      .on("error", (error: Error) => reject(error))
      .on("end", () => {
        resolve({ totalRows, skippedRows, classDistribution, sample });
      });
  });
};
