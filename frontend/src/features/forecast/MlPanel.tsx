import type { MlResponse } from "../../api/client";
import { Card } from "../../components/ui";

const pct = (n: number) => `${Math.round(n * 100)}%`;

export default function MlPanel({ ml }: { ml: MlResponse }) {
  const m = ml.metrics;
  const rows = m && [
    { label: "Recent months (test)", s: m.test_recent },
    { label: "Earlier period (validation)", s: m.validation },
  ];
  return (
    <div className="grid g2">
      <Card title="How the ML risk model performs" sub={`Will a SKU run out in ${m?.horizon_days ?? 30} days? · versus the old days-of-stock rule`}>
        {rows ? (
          <>
            <table className="cmp">
              <thead>
                <tr><th>Period</th><th>ML average precision</th><th>Rule</th><th>ML recall</th><th>Rule recall</th></tr>
              </thead>
              <tbody>
                {rows.map(({ label, s }) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className={s.ml.pr_auc > s.rule.pr_auc ? "win" : ""}>{s.ml.pr_auc.toFixed(2)}</td>
                    <td>{s.rule.pr_auc.toFixed(2)}</td>
                    <td>{pct(s.ml.at_same_flag_count.recall)}</td>
                    <td>{pct(s.rule.at_30_days.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="tiny muted" style={{ margin: "10px 0 0" }}>
              Average precision: higher is better. Recall compares both methods flagging the same number of SKU-weeks.
              Trained on a simulated stock history, so this shows the method works, not real-world accuracy.
            </p>
          </>
        ) : <p className="muted small">Model metrics not found. Run <code>python ml_risk.py</code>.</p>}
      </Card>
      <Card title="Pattern across at-risk SKUs" sub={ml.insight?.source && ml.insight.source !== "template" ? `Written by ${ml.insight.source}` : "Summary"}>
        {ml.insight ? <p className="insight">{ml.insight.text}</p>
          : <p className="muted small">No summary yet. Run <code>python explain.py</code>.</p>}
        <p className="tiny muted" style={{ margin: "10px 0 0" }}>The language model only words the findings. Every number comes from the ML model.</p>
      </Card>
    </div>
  );
}
