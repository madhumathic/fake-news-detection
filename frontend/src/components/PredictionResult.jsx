export default function PredictionResult({ result }) {
  if (!result) return null;
  const real = result.label === "REAL";
  const reliability = result.reliability || {};
  return (
    <section className={`card result ${real ? "result-real" : "result-fake"}`} aria-live="polite">
      <div className="verdict-row">
        <div className="verdict-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="30" height="30">
            {real ? (
              <path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            )}
          </svg>
        </div>
        <div>
          <p className="verdict-label">Prediction:</p>
          <h2 className="verdict">{result.prediction}</h2>
        </div>
      </div>

      <p className="confidence">Confidence: <strong>{result.confidence.toFixed(2)}%</strong></p>
      <div className="meter" role="img" aria-label={`Confidence ${result.confidence.toFixed(2)} percent`}>
        <span style={{ width: `${result.confidence}%` }} />
      </div>

      {reliability.message && (
        <p className={`reliability reliability-${reliability.level}`}>{reliability.message}</p>
      )}
      <p className="meta">{result.words} words analysed. Saved to your history.</p>
      <p className="disclaimer">{result.disclaimer}</p>
    </section>
  );
}
