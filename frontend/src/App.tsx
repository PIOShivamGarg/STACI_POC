import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import ForecastPage from "./features/forecast/ForecastPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<ForecastPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
