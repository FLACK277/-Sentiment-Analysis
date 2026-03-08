import { execFile } from "node:child_process";
import { access, mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { AnalysisResponse, ModelMetadata } from "../domain/types.js";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../../..");
const artifactDir = path.resolve(__dirname, "../../ml/artifacts");
const metadataPath = path.join(artifactDir, "notebook_model_metadata.json");
const trainScriptPath = path.resolve(__dirname, "../../ml/train_model.py");
const predictScriptPath = path.resolve(__dirname, "../../ml/predict.py");

let metadataCache: ModelMetadata | null = null;

const parseJsonOutput = <T>(stdout: string, context: string): T => {
  const payload = stdout.trim();
  if (!payload) {
    throw new Error(`${context} returned an empty response.`);
  }

  try {
    return JSON.parse(payload) as T;
  } catch {
    throw new Error(`${context} returned invalid JSON.`);
  }
};

const resolvePythonExecutable = async (): Promise<string> => {
  const configured = process.env.PYTHON_EXECUTABLE ?? path.resolve(repoRoot, ".venv/bin/python");

  try {
    await access(configured);
    return configured;
  } catch {
    return "python3";
  }
};

const runPythonScript = async <T>(scriptPath: string, args: string[], context: string): Promise<T> => {
  const pythonExecutable = await resolvePythonExecutable();

  try {
    const { stdout, stderr } = await execFileAsync(pythonExecutable, [scriptPath, ...args], {
      cwd: repoRoot,
      maxBuffer: 10 * 1024 * 1024
    });

    if (stderr.trim()) {
      console.warn(stderr.trim());
    }

    return parseJsonOutput<T>(stdout, context);
  } catch (error) {
    const stderr = typeof error === "object" && error !== null && "stderr" in error
      ? String(error.stderr ?? "")
      : "";
    const message = stderr.trim() || (error instanceof Error ? error.message : `${context} failed.`);
    throw new Error(message);
  }
};

const loadCachedMetadata = async (): Promise<ModelMetadata | null> => {
  if (metadataCache) {
    return metadataCache;
  }

  try {
    const raw = await readFile(metadataPath, "utf8");
    metadataCache = JSON.parse(raw) as ModelMetadata;
    return metadataCache;
  } catch {
    return null;
  }
};

export const trainSentimentModel = async (datasetPath: string): Promise<ModelMetadata> => {
  await mkdir(artifactDir, { recursive: true });

  const metadata = await runPythonScript<ModelMetadata>(
    trainScriptPath,
    ["--dataset", datasetPath, "--artifact-dir", artifactDir],
    "Notebook model training"
  );

  metadataCache = metadata;
  return metadata;
};

export const getModelMetadata = (): ModelMetadata => {
  if (!metadataCache) {
    throw new Error("Model is not initialized yet.");
  }

  return metadataCache;
};

export const analyzeText = async (input: string): Promise<AnalysisResponse> => {
  const text = input?.trim();
  if (!text) {
    throw new Error("Input text is required.");
  }

  const cachedMetadata = await loadCachedMetadata();
  if (!cachedMetadata) {
    throw new Error("Model not initialized. Call trainSentimentModel before analyzeText.");
  }

  return runPythonScript<AnalysisResponse>(
    predictScriptPath,
    ["--artifact-dir", artifactDir, "--text", text],
    "Notebook model inference"
  );
};
