import { useEffect, useState } from "react";
import { figureUrl, getDatasetSummary } from "../services/api.js";

const PHASES = [
  ["1. Problem definition", "Decide what to predict (REAL or FAKE), for whom, and what counts as success."],
  ["2. Requirement analysis", "List functional and non-functional requirements, software and hardware needs."],
  ["3. Data acquisition", "Load the Kaggle dataset, inspect its columns and labels, and check its quality."],
  ["4. Data preparation", "Remove empty, short and duplicate articles; clean HTML, links and the agency tag; split 80/20."],
  ["5. Exploratory data analysis", "Plot class balance, text length, subjects and frequent words."],
  ["6. Model selection", "Compare Logistic Regression, Naive Bayes, Linear SVM and Random Forest with cross-validation."],
  ["7. Model development", "Tune the winner, train it on the training set only and save the model and TF-IDF vectorizer."],
  ["8. Model evaluation", "Test once on unseen articles; study the errors; run sanity checks."],
  ["9. Integration and testing", "Serve the model through a Flask API and a React app; test everything."],
];

export default function Methodology() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getDatasetSummary().then(setData).catch((e) => setError(e.message));
  }, []);

  const report = data?.data_report;
  const steps = report?.cleaning_steps;
  return (
    <>
      <section className="page-head">
        <h1>Methodology</h1>
        <p className="subtitle">How the system was built, following the 9-phase AI project lifecycle.</p>
      </section>
      {error && <div className="alert alert-error" role="alert">{error}</div>}

      <section className="card">
        <h2>AI lifecycle</h2>
        <ol className="plain-list">
          {PHASES.map(([title, text]) => (
            <li key={title}><strong>{title}:</strong> {text}</li>
          ))}
        </ol>
      </section>

      {report && (
        <section className="card">
          <h2>Dataset used</h2>
          <p>
            {report.files.map((f) => f.file).join(", ")}: {report.records.toLocaleString()} records, {report.n_columns} columns
            ({Object.keys(report.columns).join(", ")}). Class counts: {Object.entries(report.class_distribution).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(", ")}.
          </p>
          {steps && (
            <p>
              Cleaning kept {steps.rows_out.toLocaleString()} of {steps.rows_in.toLocaleString()} records
              (removed {steps.removed_empty_text} empty, {steps.removed_too_short} too short, {steps.removed_duplicates} duplicates).
            </p>
          )}
          <h3>Leakage checks</h3>
          <ul className="plain-list">
            {report.leakage_checks.filter((c) => c.suspicious).map((c) => (
              <li key={c.column}>
                <strong>{c.column}</strong> predicts the label almost perfectly (Cram&eacute;r&apos;s V = {c.association_with_label_cramers_v}), so it is <strong>not used</strong> as a feature.
              </li>
            ))}
            {report.agency_tag_check?.suspicious && (
              <li>A news-agency tag appears mostly in REAL articles, so it is removed from the text before training and prediction.</li>
            )}
            {report.date_format_check?.suspicious && <li>The date format differs between classes, so the date is not used.</li>}
          </ul>
          <p>Only the combined <strong>title + text</strong> is used for prediction.</p>
        </section>
      )}

      <section className="card">
        <h2>Technique</h2>
        <p><strong>TF-IDF</strong> turns each article into numbers by weighting words and two-word phrases by how characteristic they are. A linear classifier then learns which weights point towards REAL or FAKE. The vectorizer is fitted on training data only, so the test data stays unseen.</p>
      </section>

      {data?.figures?.length > 0 && (
        <section className="card">
          <h2>Exploratory data analysis</h2>
          <div className="figure-grid">
            {data.figures.filter((n) => n !== "confusion_matrix.png").map((name) => (
              <img key={name} className="figure" src={figureUrl(name)} alt={name.replace(".png", "").replace(/_/g, " ")} />
            ))}
          </div>
        </section>
      )}
    </>
  );
}
