import { NavLink, Outlet } from "react-router-dom";

export default function AppShell() {
  return (
    <>
      <header className="nav">
        <div className="nav-in">
          <span className="logo">Staci</span>
          <nav aria-label="Main">
            <NavLink to="/" end>Forecasting</NavLink>
          </nav>
        </div>
      </header>
      <main className="page"><Outlet /></main>
    </>
  );
}
