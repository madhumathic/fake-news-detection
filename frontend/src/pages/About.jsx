import { Link } from "react-router-dom";

export default function About() {
  return (
    <>
      <section className="page-head">
        <h1>About this project</h1>
        <p className="subtitle">What the system does and where it can go wrong.</p>
      </section>
      <section className="card prose">
        <h2>What it does</h2>
        <p>You paste a news article and the system predicts <strong>REAL NEWS</strong> or <strong>FAKE NEWS</strong> together with a confidence value. Every prediction is stored in a SQLite history table.</p>
        <h2>How to read a prediction</h2>
        <p>The prediction is based on <strong>patterns learned from the training dataset</strong>, mainly vocabulary and writing style. The model does not check facts, search the internet or know about current events. It is an ML prediction system, not a fact-checking system.</p>
        <h2>Why predictions can be wrong</h2>
        <ul>
          <li><strong>Different from the training data:</strong> the model learned from a particular set of sources and period. Articles that look different (other topics, countries, publishers or years) are out-of-distribution and the model may be unreliable on them.</li>
          <li><strong>Style is not truth:</strong> a well-written false story can look real, and a clumsy true story can look fake.</li>
          <li><strong>Confidence is not correctness:</strong> a high confidence means the model is sure, not that it is right.</li>
          <li><strong>Language and length:</strong> only English is supported and very short text gives weak results.</li>
        </ul>
        <p>Always confirm important claims with several trusted sources.</p>
      </section>
      <p className="back-row"><Link className="btn btn-ghost" to="/">Back to Home</Link></p>
    </>
  );
}
