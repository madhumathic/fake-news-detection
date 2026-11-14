export default function ConfusionMatrix({ matrix, labels = ["FAKE", "REAL"] }) {
  if (!matrix) return null;
  const max = Math.max(...matrix.flat());
  return (
    <div className="cm" role="table" aria-label="Confusion matrix">
      <div className="cm-corner" />
      <div className="cm-head">Predicted {labels[0]}</div>
      <div className="cm-head">Predicted {labels[1]}</div>
      {matrix.map((row, i) => (
        <div className="cm-row" role="row" key={i}>
          <div className="cm-head cm-side">Actual {labels[i]}</div>
          {row.map((value, j) => (
            <div
              role="cell"
              key={j}
              className={`cm-cell ${i === j ? "cm-good" : "cm-bad"}`}
              style={{ opacity: 0.35 + 0.65 * (value / (max || 1)) }}
            >
              {value.toLocaleString()}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
