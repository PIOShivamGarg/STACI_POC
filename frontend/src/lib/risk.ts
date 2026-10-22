/** Risk levels come from the API (risk.py). The UI only maps them to colours. */
export type RiskLevel = "High risk" | "Watch" | "Monitor" | "Healthy" | "Possible overstock";
export type FvaStatus = "On track" | "Watch" | "Off track";

export const AT_RISK: RiskLevel[] = ["High risk", "Watch", "Monitor"];

export const LEVEL_PILL: Record<RiskLevel, string> = {
  "High risk": "pill-crit",
  Watch: "pill-warn",
  Monitor: "pill-info",
  Healthy: "pill-ok",
  "Possible overstock": "pill-neutral",
};

export const FVA_PILL: Record<FvaStatus, string> = {
  "On track": "pill-ok",
  Watch: "pill-warn",
  "Off track": "pill-crit",
};

/** Wording that depends on how levels are set: ML probability or days-of-stock rules. */
export const riskCopy = (method?: "ml" | "rules") =>
  method === "ml"
    ? { shortfall: "25%+ chance of running out in 30 days", high: "50%+ chance of running out", under30: "SKUs at shortfall risk" }
    : { shortfall: "Under 30 days of stock", high: "Under 14 days of stock", under30: "SKUs under 30 days of stock" };
