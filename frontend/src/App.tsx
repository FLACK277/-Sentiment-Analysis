import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { SentimentScene } from "./components/SentimentScene";
import { useSentiment } from "./hooks/useSentiment";

const initialText = "Battery lasted 2 days and performance stayed excellent throughout.";

const App = () => {
  const [text, setText] = useState(initialText);
  const { result, isLoading, error, analyze } = useSentiment();

  const confidencePercent = useMemo(() => Math.round((result?.confidence ?? 0) * 100), [result]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await analyze(text);
  };

  return (
    <main className="layout">
      <section className="panel">
        <h1>3D Sentiment Studio</h1>
        <p className="subtitle">Type laptop feedback, run analysis, and inspect sentiment in an interactive 3D panel.</p>

        <form onSubmit={onSubmit} className="input-form">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Enter review text..."
            rows={5}
          />
          <button disabled={isLoading || text.trim().length === 0} type="submit">
            {isLoading ? "Analyzing..." : "Analyze sentiment"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <motion.div
            className="result"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
          >
            <p><strong>Sentiment:</strong> {result.sentiment}</p>
            <p><strong>Confidence:</strong> {confidencePercent}%</p>
            <div className="confidence-track" aria-hidden>
              <div className="confidence-fill" style={{ width: `${confidencePercent}%` }} />
            </div>
            <p><strong>Score:</strong> {result.score}</p>
            <p><strong>Top tokens:</strong> {result.tokens.slice(0, 12).join(", ") || "n/a"}</p>
            <ul className="contributors">
              {result.topContributors.map((entry) => (
                <li key={`${entry.token}-${entry.weight}`}>
                  <span>{entry.token}</span>
                  <span>{entry.weight > 0 ? `+${entry.weight}` : entry.weight}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </section>

      <section className="viewer">
        <SentimentScene result={result} loading={isLoading} />
      </section>
    </main>
  );
};

export default App;
