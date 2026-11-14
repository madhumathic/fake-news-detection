import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import HistoryTable from "../components/HistoryTable.jsx";
import { clearHistory, getHistory } from "../services/api.js";

export default function History() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setError("");
      setData(await getHistory());
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const clearAll = async () => {
    if (!window.confirm("Delete all saved predictions? This cannot be undone.")) return;
    try {
      await clearHistory();
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  const total = data?.total ?? 0;
  return (
    <>
      <section className="page-head">
        <h1>Prediction history</h1>
        <p className="subtitle">
          {!data ? "Loading..." : total === 0 ? "Articles you check will be listed here." : `${total} saved prediction${total === 1 ? "" : "s"}.`}
        </p>
      </section>
      {error && <div className="alert alert-error" role="alert">{error}</div>}
      <section className="card">
        {data && total > 0 && <HistoryTable items={data.items} />}
        {data && total === 0 && <p className="empty">No predictions yet. Check an article and it will appear here.</p>}
      </section>
      <div className="history-actions">
        <Link to="/" className="btn btn-ghost">Back to Home</Link>
        {total > 0 && <button className="btn btn-danger" onClick={clearAll}>Clear All History</button>}
      </div>
    </>
  );
}
