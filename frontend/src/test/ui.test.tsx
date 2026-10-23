import { fireEvent, render, screen } from "@testing-library/react";
import type { RiskRow } from "../api/client";
import ShortfallTable, { asnStatus, chanceText, runsOut } from "../features/forecast/ShortfallTable";
import { buildSeries, peakInfo } from "../features/forecast/volumeData";
import { fmtDate, signed } from "../lib/format";

const row = (o: Partial<RiskRow>): RiskRow => ({
  sku: "CL01-SKU001", client_id: "CL01", client_name: "Northwind", category: "Tops", description: "Tops item 1",
  warehouse: "Dallas", available_qty: 10, avg_daily_usage_next4w: 2, days_of_cover: 5, lead_time_days: 7,
  covered_by_asn: null, next_asn_date: null, asn_arrives_in_time: false, projected_stockout_p50: "2026-10-17",
  projected_stockout_p90: "2026-10-16", risk_state: "High risk", reason: "", suggested_reorder_qty: 50, ...o,
});

const counts = { "High risk": 1, Watch: 1, Monitor: 1, Healthy: 0, "Possible overstock": 0 };

describe("format", () => {
  it("formats an ISO date without shifting the day", () => expect(fmtDate("2026-10-29")).toBe("Oct 29"));
  it("signs percentages", () => {
    expect(signed(2.4)).toBe("+2.4%");
    expect(signed(-7.9)).toBe("-7.9%");
  });
});

describe("volume series", () => {
  const v = {
    actual: [{ week_start: "2026-10-05", actual: 100 }],
    forecast: [{ week_start: "2026-10-12", p50: 120, p90: 150 }, { week_start: "2026-10-19", p50: 150, p90: 190 }],
  };
  it("starts the forecast line at the last actual point", () => {
    const s = buildSeries(v);
    expect(s).toHaveLength(3);
    expect(s[0].forecast).toBe(100);
    expect(s[1].range).toEqual([120, 150]);
  });
  it("finds the peak and its change vs the latest week", () =>
    expect(peakInfo(v)).toMatchObject({ week: "2026-10-19", orders: 150, pctVsLatest: 50 }));
});

describe("ASN status", () => {
  it("flags a missing ASN", () => expect(asnStatus(row({})).tone).toBe("crit"));
  it("shows covered when the ASN lands in time", () => {
    const s = asnStatus(row({ covered_by_asn: "ASN1", next_asn_date: "2026-10-20", asn_arrives_in_time: true }));
    expect(s).toEqual({ text: "Covered by ASN1 (ETA Oct 20)", tone: "ok" });
  });
  it("shows the likely date when the supplier usually runs late", () => {
    const s = asnStatus(row({ covered_by_asn: "ASN1", next_asn_date: "2026-10-20", next_asn_likely_date: "2026-10-25" }));
    expect(s).toEqual({ text: "ASN1 (ETA Oct 20, likely Oct 25) — after stockout", tone: "warn" });
  });
  it("warns when the ASN is too late", () =>
    expect(asnStatus(row({ covered_by_asn: "ASN1", next_asn_date: "2026-10-29" })).tone).toBe("warn"));
  it("shows the overstock chance on overstock rows", () => {
    expect(chanceText(row({ risk_state: "Possible overstock", risk_probability: 0.01, overstock_probability: 0.74 }))).toBe("74% chance");
    expect(chanceText(row({ risk_probability: 0.88, overstock_probability: 0.01 }))).toBe("88% chance");
  });
  it("handles no projected stockout", () => expect(runsOut(row({ projected_stockout_p50: null }))).toMatch(/No stockout/));
});

describe("ShortfallTable", () => {
  const rows = [
    row({ sku: "A", description: "Alpha", risk_state: "High risk" }),
    row({ sku: "B", description: "Bravo", risk_state: "Watch" }),
    row({ sku: "C", description: "Charlie", risk_state: "Monitor" }),
    row({ sku: "D", description: "Delta", risk_state: "Healthy" }),
  ];
  it("lists only at-risk levels", () => {
    render(<ShortfallTable rows={rows} counts={counts} />);
    expect(screen.getByText("Alpha")).toBeInTheDocument();
    expect(screen.queryByText("Delta")).not.toBeInTheDocument();
  });
  it("filters by level", () => {
    render(<ShortfallTable rows={rows} counts={counts} />);
    fireEvent.click(screen.getByRole("button", { name: /^Watch/ }));
    expect(screen.getByText("Bravo")).toBeInTheDocument();
    expect(screen.queryByText("Alpha")).not.toBeInTheDocument();
  });
  it("only offers Create ASN when no ASN exists", () => {
    render(<ShortfallTable rows={[row({ covered_by_asn: "ASN1", description: "Covered" })]} counts={counts} />);
    expect(screen.queryByRole("button", { name: "Create ASN" })).not.toBeInTheDocument();
  });
});

describe("ML risk details", () => {
  const r = row({ risk_probability: 0.91, top_drivers: "only 8 days of stock; next shipment due in 25 days",
                  explanation: "Likely to run out soon.", rule_risk_state: "Watch" });
  it("shows the probability and reveals the explanation on Why?", () => {
    render(<ShortfallTable rows={[r]} counts={counts} />);
    expect(screen.getByText("91% chance")).toBeInTheDocument();
    expect(screen.queryByText("Likely to run out soon.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Why?" }));
    expect(screen.getByText("Likely to run out soon.")).toBeInTheDocument();
    expect(screen.getByText("only 8 days of stock")).toBeInTheDocument();
    expect(screen.getByText(/old days-of-stock rule said Watch/)).toBeInTheDocument();
  });
});
