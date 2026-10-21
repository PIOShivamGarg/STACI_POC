import { createContext, useContext, useState, type ReactNode } from "react";
import type { Filters } from "../api/client";

const Ctx = createContext<{ filters: Filters; setFilters: (f: Filters) => void } | null>(null);

export function FilterProvider({ children }: { children: ReactNode }) {
  const [filters, setFilters] = useState<Filters>({ model: "lgbm", warehouse: "" });
  return <Ctx.Provider value={{ filters, setFilters }}>{children}</Ctx.Provider>;
}

export function useFilters() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useFilters must be used inside FilterProvider");
  return c;
}
