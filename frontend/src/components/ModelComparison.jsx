const pct = (v) => `${(v * 100).toFixed(2)}%`;

export default function ModelComparison({ rows }) {
  if (!rows || rows.length === 0) return <p>No comparison available yet. Run <code>python scripts/train_models.py</code>.</p>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Model</th><th>CV accuracy</th><th>CV precision</th><th>CV recall</th><th>CV F1</th><th>Train accuracy</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model} className={r.selected ? "row-selected" : ""}>
              <td>
                {r.model}
                {r.selected && <span className="badge badge-real">selected</span>}
                {r.is_baseline && <span className="badge badge-muted">baseline</span>}
              </td>
              <td>{pct(r.cv_accuracy_mean)} <small>&plusmn;{(r.cv_accuracy_std * 100).toFixed(2)}</small></td>
              <td>{pct(r.cv_precision_mean)}</td>
              <td>{pct(r.cv_recall_mean)}</td>
              <td>{pct(r.cv_f1_mean)}</td>
              <td>{pct(r.train_accuracy_mean)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
