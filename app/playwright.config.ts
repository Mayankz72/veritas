import { defineConfig } from "@playwright/test";

/**
 * E2E tests assume the Veritas ml-service is already running externally
 * (docker compose up -d && uvicorn app.main:app --port 8000) - Playwright
 * only manages the Next.js dev server itself, not the Python backend or
 * Postgres.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:4173",
    trace: "retain-on-failure",
    video: process.env.RECORD_DEMO ? "on" : "retain-on-failure",
    viewport: { width: 1280, height: 800 },
  },
  webServer: {
    command: "npx next dev -p 4173",
    url: "http://localhost:4173",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
