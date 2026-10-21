import { NavLink, Outlet } from "react-router-dom";

const SOON = ["Orders", "Receipts & ASNs", "Inventory", "Returns", "Reports"];

export default function AppShell() {
  return (
    <>
      <header className="nav">
        <div className="nav-in">
          <span className="logo">Staci</span>
          <nav aria-label="Main">
            <NavLink to="/" end>Home</NavLink>
            {SOON.map((n) => (
              <span key={n} className="nav-off" title="Not part of this prototype" aria-disabled="true">{n}</span>
            ))}
            <NavLink to="/forecast">Forecasting</NavLink>
          </nav>
        </div>
      </header>
      <main className="page"><Outlet /></main>
    </>
  );
}
