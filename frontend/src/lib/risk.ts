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
