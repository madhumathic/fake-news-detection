// All communication with the Flask backend. Paths are relative ("/api/...") so the
// Vite dev proxy (or the Flask server itself, after "npm run build") answers them.
const BASE = import.meta.env.VITE_API_URL || "https://fake-news-detection-s6ig.onrender.com";

export class ApiError extends Error {
  constructor(message, code) {
    super(message);
    this.code = code;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError(
      "Cannot reach the server. Make sure the backend is running (python -m app.main).",
      "network_error"
    );
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    /* the body was not JSON */
  }
  if (!response.ok) {
    throw new ApiError(
      data?.error?.message || `The server returned an error (${response.status}).`,
      data?.error?.code || "http_error"
    );
  }
  return data;
}

export const predictNews = (news) =>
  request("/api/predict", { method: "POST", body: JSON.stringify({ news }) });
export const getHistory = (limit = 100) => request(`/api/history?limit=${limit}`);
export const clearHistory = () => request("/api/history", { method: "DELETE" });
export const getPerformance = () => request("/api/model/performance");
export const getDatasetSummary = () => request("/api/dataset/summary");
export const getHealth = () => request("/api/health");
export const figureUrl = (name) => `${BASE}/api/analysis/figures/${name}`;
