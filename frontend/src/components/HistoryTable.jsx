export default function HistoryTable({ items }) {
  return (
    <div className="table-wrap">
      <table className="table history-table">
        <thead>
          <tr><th>News</th><th>Prediction</th><th>Confidence</th><th>Date</th></tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id}>
              <td data-label="News">
                {item.preview !== item.news ? (
                  <details>
                    <summary>{item.preview} <span className="more">Show full text</span></summary>
                    <p className="full-text">{item.news}</p>
                  </details>
                ) : (
                  item.preview
                )}
              </td>
              <td data-label="Prediction">
                <span className={`badge ${item.prediction === "REAL NEWS" ? "badge-real" : "badge-fake"}`}>{item.prediction}</span>
              </td>
              <td data-label="Confidence">{item.confidence.toFixed(2)}%</td>
              <td data-label="Date" className="nowrap">{item.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
