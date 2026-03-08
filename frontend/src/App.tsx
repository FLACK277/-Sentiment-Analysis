import { FormEvent, useEffect, useMemo, useState } from "react";
import { useSentiment } from "./hooks/useSentiment";
import type { HealthResponse, SentimentLabel } from "./types/sentiment";

const initialText = "Battery lasted 2 days and performance stayed excellent throughout.";

const samplePrompts = [
  "Battery backup stayed solid through two workdays and the keyboard feels premium.",
  "The laptop overheats fast, lags during meetings, and the trackpad feels cheap.",
  "Display quality is fine and performance is acceptable, but nothing stands out."
];

const sentimentStyles: Record<SentimentLabel, { badge: string; ring: string; accent: string }> = {
  positive: {
    badge: "bg-emerald-500/15 text-emerald-200 ring-1 ring-inset ring-emerald-400/30",
    ring: "ring-emerald-400/30",
    accent: "from-emerald-400/80 to-teal-300/80"
  },
  negative: {
    badge: "bg-rose-500/15 text-rose-200 ring-1 ring-inset ring-rose-400/30",
    ring: "ring-rose-400/30",
    accent: "from-rose-400/80 to-orange-300/80"
  },
  neutral: {
    badge: "bg-sky-500/15 text-sky-200 ring-1 ring-inset ring-sky-400/30",
    ring: "ring-sky-400/30",
    accent: "from-sky-400/80 to-cyan-300/80"
  }
};

const apiBaseUrl = import.meta.env.VITE_API_URL ?? "";

const parseJson = async <T,>(response: Response): Promise<T> => {
  const text = await response.text();
  if (!text.trim()) {
    throw new Error("The server returned an empty response.");
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error("The backend returned invalid JSON.");
  }
};

const toStars = (sentiment: SentimentLabel): number => {
  if (sentiment === "positive") return 5;
  if (sentiment === "negative") return 2;
  return 3;
};

const compactProduct = (name: string): string => {
  const trimmed = name.replace(/\.\.\.$/, "").trim();
  return trimmed.length > 72 ? `${trimmed.slice(0, 72)}...` : trimmed;
};

const App = () => {
  const [text, setText] = useState(initialText);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const { result, isLoading, error, analyze } = useSentiment();

  const confidencePercent = useMemo(() => Math.round((result?.confidence ?? 0) * 100), [result]);
  const currentSentiment = result?.sentiment ?? "neutral";
  const sentimentStyle = sentimentStyles[currentSentiment];

  useEffect(() => {
    let cancelled = false;

    const loadHealth = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/health`);
        const payload = await parseJson<HealthResponse | { error?: string }>(response);

        if (!response.ok) {
          throw new Error("error" in payload ? payload.error ?? "Failed to load API status" : "Failed to load API status");
        }

        if (!cancelled) {
          setHealth(payload as HealthResponse);
          setHealthError(null);
        }
      } catch (caught) {
        if (!cancelled) {
          setHealthError(caught instanceof Error ? caught.message : "Failed to load API status");
        }
      }
    };

    void loadHealth();

    return () => {
      cancelled = true;
    };
  }, []);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await analyze(text);
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(42,193,188,0.18),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(244,114,182,0.12),_transparent_24%),linear-gradient(180deg,_#07111f_0%,_#091827_46%,_#050b14_100%)] text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <header className="rounded-[1.75rem] border border-white/10 bg-slate-950/65 px-5 py-4 shadow-[0_18px_55px_rgba(2,8,23,0.4)] backdrop-blur-xl sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="grid size-10 place-items-center rounded-2xl bg-gradient-to-br from-cyan-400 to-sky-300 text-slate-900 shadow-[0_10px_24px_rgba(34,211,238,0.35)]">R</span>
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-200">Review Harbor</p>
                <p className="text-xs text-slate-400">Laptop buyer opinions and sentiment signals</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="rounded-full bg-white/5 px-3 py-1 text-slate-300 ring-1 ring-inset ring-white/10">
                {health?.status === "ok" ? "API Online" : "Checking API"}
              </span>
              <span className="rounded-full bg-white/5 px-3 py-1 text-slate-300 ring-1 ring-inset ring-white/10">
                {health?.model.modelName ?? "Model loading"}
              </span>
              <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-cyan-200 ring-1 ring-inset ring-cyan-400/30">
                {health ? `${Math.round(health.model.validationAccuracy * 100)}% accuracy` : "Training..."}
              </span>
            </div>
          </div>
        </header>

        <section className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
          <div className="space-y-6">
            <section className="rounded-[1.75rem] border border-white/10 bg-slate-950/60 p-6 shadow-[0_18px_55px_rgba(2,8,23,0.36)] backdrop-blur-xl sm:p-7">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h1 className="font-[Space_Grotesk] text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    Latest Laptop Reviews
                  </h1>
                  <p className="mt-2 text-sm text-slate-300">
                    Browse real review snippets and run quick sentiment checks like a product review portal.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-300">
                  <span className="rounded-full bg-white/5 px-3 py-1 ring-1 ring-inset ring-white/10">All brands</span>
                  <span className="rounded-full bg-white/5 px-3 py-1 ring-1 ring-inset ring-white/10">Most recent</span>
                  <span className="rounded-full bg-white/5 px-3 py-1 ring-1 ring-inset ring-white/10">Verified buyers</span>
                </div>
              </div>

              <div className="mt-6 space-y-4">
                {(health?.dataset.sample ?? []).map((sample, index) => {
                  const stars = toStars(sample.sentiment);
                  return (
                    <article key={`${sample.product}-${index}`} className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 transition hover:bg-white/[0.07]">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <h2 className="text-base font-semibold text-white">{sample.title || "Untitled review"}</h2>
                        <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] ${sentimentStyles[sample.sentiment].badge}`}>
                          {sample.sentiment}
                        </span>
                      </div>
                      <p className="mt-2 text-sm text-slate-300">{compactProduct(sample.product)}</p>
                      <div className="mt-4 flex items-center justify-between">
                        <p className="text-sm tracking-[0.3em] text-amber-300">{"★".repeat(stars)}{"☆".repeat(5 - stars)}</p>
                        <button
                          type="button"
                          onClick={() => setText(sample.title || sample.product)}
                          className="rounded-full border border-cyan-300/25 bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-500/20"
                        >
                          Analyze this review
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>

              {!health?.dataset.sample.length && (
                <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
                  Review feed will appear once dataset samples are loaded.
                </div>
              )}
            </section>

            <section className="rounded-[1.75rem] border border-white/10 bg-slate-950/60 p-6 shadow-[0_18px_55px_rgba(2,8,23,0.36)] backdrop-blur-xl">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm text-slate-400">Marketplace Snapshot</p>
                  <h3 className="mt-1 font-[Space_Grotesk] text-xl font-semibold text-white">Community sentiment trend</h3>
                </div>
                <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300 ring-1 ring-inset ring-white/10">
                  {health?.model.trainedRows.toLocaleString() ?? "..."} reviews indexed
                </span>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-3">
                <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-emerald-200">Positive</p>
                  <p className="mt-2 text-2xl font-semibold text-white">{health?.dataset.classDistribution.positive.toLocaleString() ?? "..."}</p>
                </div>
                <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-rose-200">Negative</p>
                  <p className="mt-2 text-2xl font-semibold text-white">{health?.dataset.classDistribution.negative.toLocaleString() ?? "..."}</p>
                </div>
                <div className="rounded-2xl border border-cyan-400/20 bg-cyan-500/10 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-cyan-200">Model source</p>
                  <p className="mt-2 text-sm font-semibold text-white">{health?.model.modelSource ?? "LaptopML.ipynb"}</p>
                </div>
              </div>
            </section>
          </div>

          <div className="space-y-6 xl:sticky xl:top-8 xl:self-start">
            <section className="rounded-[1.75rem] border border-white/10 bg-slate-950/65 p-6 shadow-[0_18px_55px_rgba(2,8,23,0.4)] backdrop-blur-xl sm:p-7">
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="font-[Space_Grotesk] text-2xl font-semibold text-white">Write a review</h2>
                <span className="rounded-full bg-white/5 px-3 py-1 text-xs font-medium text-slate-300 ring-1 ring-inset ring-white/10">
                  Instant sentiment scan
                </span>
              </div>

              <form className="mt-6 space-y-4" onSubmit={onSubmit}>
              <label className="block text-sm font-medium text-slate-300" htmlFor="review-text">
                Review text
              </label>
              <textarea
                id="review-text"
                className="min-h-48 w-full rounded-[1.5rem] border border-white/10 bg-slate-900/80 px-4 py-4 text-base text-slate-100 shadow-inner outline-none transition focus:border-cyan-300/40 focus:ring-4 focus:ring-cyan-400/10"
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Describe the laptop experience you want to classify..."
              />

              <div className="flex flex-wrap gap-2">
                {samplePrompts.map((prompt) => (
                  <button
                    key={prompt}
                    className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-left text-xs text-slate-300 transition hover:border-cyan-300/30 hover:bg-cyan-400/10 hover:text-white"
                    type="button"
                    onClick={() => setText(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <button
                  className="inline-flex items-center justify-center rounded-full bg-[linear-gradient(135deg,_#2dd4bf_0%,_#38bdf8_45%,_#f472b6_100%)] px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_10px_35px_rgba(45,212,191,0.28)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={isLoading || text.trim().length === 0}
                  type="submit"
                >
                  {isLoading ? "Analyzing review..." : "Run sentiment analysis"}
                </button>
                <p className="text-sm text-slate-400">The first request may take longer if the notebook model is still training.</p>
              </div>
              </form>

              {error && (
                <div className="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              )}

              {healthError && (
                <div className="mt-4 rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                  {healthError}
                </div>
              )}
            </section>

            <section className={`rounded-[2rem] border bg-slate-950/60 p-6 shadow-[0_18px_60px_rgba(2,8,23,0.4)] backdrop-blur-xl ring-1 ${sentimentStyle.ring}`}>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="text-sm text-slate-400">Latest prediction</p>
                  <h2 className="mt-1 font-[Space_Grotesk] text-2xl font-semibold text-white">
                    {result ? result.sentiment : "Awaiting analysis"}
                  </h2>
                </div>
                <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${sentimentStyle.badge}`}>
                  {result ? `${confidencePercent}% confidence` : "Ready"}
                </span>
              </div>

              <div className="mt-6 h-3 overflow-hidden rounded-full bg-white/10">
                <div
                  className={`h-full rounded-full bg-[linear-gradient(90deg,var(--tw-gradient-stops))] ${sentimentStyle.accent}`}
                  style={{ width: `${result ? confidencePercent : 18}%` }}
                />
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <p className="text-sm text-slate-400">Score</p>
                  <p className="mt-2 text-3xl font-semibold text-white">{result ? result.score.toFixed(3) : "0.000"}</p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <p className="text-sm text-slate-400">Explanation</p>
                  <p className="mt-2 text-sm leading-6 text-slate-200">
                    {result?.explanation ?? "Run the API to see notebook-model output and top weighted terms."}
                  </p>
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm font-medium text-slate-300">Processed tokens</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {(result?.tokens.length ? result.tokens : ["awaiting", "analysis"]).slice(0, 18).map((token) => (
                    <span key={token} className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200">
                      {token}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm font-medium text-slate-300">Top weighted terms</p>
                <div className="mt-3 space-y-3">
                  {(result?.topContributors.length ? result.topContributors : health?.model.topFeatures ?? []).slice(0, 6).map((entry) => (
                    <div key={`${entry.token}-${entry.weight}`} className="rounded-2xl border border-white/10 bg-white/5 p-3">
                      <div className="flex items-center justify-between gap-3 text-sm text-slate-200">
                        <span className="font-medium text-white">{entry.token}</span>
                        <span>{entry.weight.toFixed(3)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </div>
        </section>
      </div>
    </main>
  );
};

export default App;
