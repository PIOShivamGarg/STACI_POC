import type { ReactNode } from "react";

export const Pill = ({ tone, children }: { tone: string; children: ReactNode }) => (
  <span className={`pill ${tone}`}>{children}</span>
);

export function Card({ title, sub, action, children, className = "" }: {
  title: string; sub?: string; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card pad ${className}`}>
      <div className="card-h">
        <h2>{title}</h2>
        {sub && <span className="sub">{sub}</span>}
        <span className="grow" />
        {action}
      </div>
      {children}
    </section>
  );
}

export function StatTile({ label, value, sub, tone }: {
  label: string; value: ReactNode; sub?: string; tone?: "crit" | "warn" | "ok";
}) {
  return (
    <div className="stat">
      <span className="lab">{label}</span>
      <span className="val" style={tone ? { color: `var(--${tone}-text)` } : undefined}>{value}</span>
      {sub && <span className="tiny muted">{sub}</span>}
    </div>
  );
}

export const Loading = ({ rows = 3 }: { rows?: number }) => (
  <div aria-busy="true" aria-label="Loading">
    {Array.from({ length: rows }, (_, i) => <div key={i} className="skeleton" />)}
  </div>
);

export const ErrorBox = ({ error }: { error: unknown }) => (
  <div className="banner banner-crit" role="alert">
    Couldn't load this data. {error instanceof Error ? error.message : ""} Check that the API is running
    (<code>uvicorn api.main:app --port 8000</code>).
  </div>
);

export const Empty = ({ children }: { children: ReactNode }) => <div className="empty">{children}</div>;
