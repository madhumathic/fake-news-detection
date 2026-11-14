import { useState } from "react";
import NewsInput from "../components/NewsInput.jsx";
import PredictionResult from "../components/PredictionResult.jsx";
import { predictNews } from "../services/api.js";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError("");
    setResult(null);
    if (!text.trim()) {
      setError("Please paste or type a news article before clicking Check News.");
      return;
    }
    setLoading(true);
    try {
      setResult(await predictNews(text));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setText("");
    setResult(null);
    setError("");
  };

  return (
    <>
      <section className="page-head">
        <h1>Fake News Detection Using Machine Learning</h1>
        <p className="subtitle">
          Paste a news article. A model trained on thousands of labelled articles estimates whether it reads like real or fake news.
        </p>
      </section>
      <NewsInput value={text} onChange={setText} onSubmit={submit} onClear={clear} loading={loading} error={error} />
      <PredictionResult result={result} />
      <section className="tips">
        <div className="tip"><h3>Include the headline</h3><p>The model was trained on headline and body together.</p></div>
        <div className="tip"><h3>Use a full article</h3><p>Longer text gives the model more writing to judge.</p></div>
        <div className="tip"><h3>It judges style, not facts</h3><p>It cannot look anything up. Check important claims with trusted sources.</p></div>
      </section>
    </>
  );
}
