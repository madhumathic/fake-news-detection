import { useEffect, useState } from "react";
import ConfusionMatrix from "../components/ConfusionMatrix.jsx";
import MetricCard from "../components/MetricCard.jsx";
import ModelComparison from "../components/ModelComparison.jsx";
import { figureUrl, getPerformance } from "../services/api.js";

const pct = (v) => `${(v * 100).toFixed(2)}%`;

export default function ModelPerformance() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getPerformance().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-error" role="alert">{error}</div>;
  if (!data) return <p>Loading...</p>;
  if (!data.available) {
    return (
      <div className="card">
        <h1>Model performance</h1>
        <p>No evaluation results found. Run <code>python scripts/train_models.py</code> and then <code>python scripts/evaluate_model.py</code>.</p>
      </div>
    );
  }

  const ev = data.evaluation;
  const meta = data.metadata || {};
  const [low, high] = ev.accuracy_95ci;
  return (
    <>
      <section className="page-head">
        <h1>Model performance</h1>
        <p className="subtitle">
          Final model: <strong>{ev.model_name}</strong>, evaluated once on {ev.test_size.toLocaleString()} articles it never saw during training or tuning.
        </p>
      </section>

      <div className={`alert reliability reliability-${data.reliability.level}`}>{data.reliability.message}</div>

      <section className="metric-grid">
        <MetricCard label="Accuracy" value={pct(ev.accuracy)} hint={`95% interval ${pct(low)} to ${pct(high)}`} />
        <MetricCard label="Precision" value={pct(ev.precision)} hint="REAL is the positive class" />
        <MetricCard label="Recall" value={pct(ev.recall)} />
        <MetricCard label="F1-score" value={pct(ev.f1_score)} />
        <MetricCard label="ROC-AUC" value={ev.roc_auc.toFixed(4)} hint="0.5 = random guessing" />
        <MetricCard label="Majority baseline" value={pct(ev.majority_baseline_accuracy)} hint="accuracy of always guessing one class" />
      </section>

      <section className="grid-2">
        <div className="card">
          <h2>Confusion matrix</h2>
          <ConfusionMatrix matrix={ev.confusion_matrix.matrix} labels={ev.confusion_matrix.labels} />
          <p className="small">
            {ev.confusion_matrix.FP_fake_as_real} fake articles were called real and {ev.confusion_matrix.FN_real_as_fake} real articles were called fake.
          </p>
        </div>
        <div className="card">
          <h2>Honest checks</h2>
          <ul className="plain-list">
            <li>Training accuracy was {pct(ev.train_accuracy)}, test accuracy is {pct(ev.accuracy)}: a gap of {pct(ev.overfit_gap_train_minus_test)} shows some overfitting.</li>
            <li>A model trained on <em>shuffled</em> labels scores {pct(ev.label_shuffle_check.mean)} here, so the real score comes from learned patterns.</li>
            <li>These scores describe articles similar to the training data. News from other sources or periods may be harder.</li>
          </ul>
        </div>
      </section>

      <section className="card">
        <h2>Model comparison (5-fold cross-validation on training data)</h2>
        <ModelComparison rows={data.comparison} />
        <p className="small">{meta.selection_note}</p>
      </section>

      {ev.top_terms && (
        <section className="grid-2">
          <div className="card"><h2>Words pushing towards FAKE</h2><p>{ev.top_terms.towards_FAKE.join(", ")}</p></div>
          <div className="card"><h2>Words pushing towards REAL</h2><p>{ev.top_terms.towards_REAL.join(", ")}</p></div>
        </section>
      )}

      <section className="card">
        <h2>Confusion matrix figure</h2>
        <img className="figure" src={figureUrl("confusion_matrix.png")} alt="Confusion matrix on the test set" />
      </section>
    </>
  );
}
