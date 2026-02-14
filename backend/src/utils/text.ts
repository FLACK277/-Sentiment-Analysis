const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this", "those", "these", "i", "you", "we", "they", "my", "our", "their", "or"
]);

export const preprocessText = (text: string): string =>
  text
    .toLowerCase()
    .replace(/<[^>]*>/g, " ")
    .replace(/[^a-z\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

export const tokenize = (text: string): string[] => {
  const clean = preprocessText(text);
  if (!clean) return [];

  return clean
    .split(" ")
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
};
