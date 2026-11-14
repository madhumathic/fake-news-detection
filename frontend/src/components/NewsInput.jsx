export default function NewsInput({ value, onChange, onSubmit, onClear, loading, error }) {
  const words = value.trim() ? value.trim().split(/\s+/).length : 0;
  const submit = (e) => {
    e.preventDefault();
    onSubmit();
  };
  return (
    <form className="card" onSubmit={submit} noValidate>
      <label htmlFor="news" className="field-label">News article (headline and body)</label>
      <textarea
        id="news"
        rows={12}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Paste or type your news article here..."
        disabled={loading}
      />
      {error && <div className="alert alert-error" role="alert">{error}</div>}
      <div className="form-foot">
        <span className="counter">{words} {words === 1 ? "word" : "words"}</span>
        <div className="actions">
          <button type="button" className="btn btn-ghost" onClick={onClear} disabled={loading}>Clear</button>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? "Checking..." : "Check News"}
          </button>
        </div>
      </div>
    </form>
  );
}
