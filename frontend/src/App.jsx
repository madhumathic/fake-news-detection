import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home.jsx";
import History from "./pages/History.jsx";
import ModelPerformance from "./pages/ModelPerformance.jsx";
import Methodology from "./pages/Methodology.jsx";
import About from "./pages/About.jsx";

const links = [
  ["/", "Home"],
  ["/history", "History"],
  ["/performance", "Model Performance"],
  ["/methodology", "Methodology"],
  ["/about", "About"],
];

export default function App() {
  return (
    <div className="app">
      <header className="site-header">
        <div className="container header-inner">
          <NavLink to="/" className="brand">
            <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
              <rect x="3" y="4" width="26" height="24" rx="5" fill="none" stroke="currentColor" strokeWidth="2" />
              <path d="M9 11h14M9 16h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              <path d="M17 22l3 3 6-7" fill="none" stroke="#6EE7B7" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>Fake News Detection</span>
          </NavLink>
          <nav className="nav" aria-label="Main navigation">
            {links.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="container main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/history" element={<History />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<div className="card"><h1>Page not found</h1><NavLink to="/" className="btn btn-primary">Back to Home</NavLink></div>} />
        </Routes>
      </main>

      <footer className="site-footer">
        <div className="container">
          Academic project. Predictions are based on patterns learned from the training dataset and are not a definitive fact-check.
        </div>
      </footer>
    </div>
  );
}
