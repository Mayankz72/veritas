import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // e2e/ holds Playwright specs (different `test`/`expect`, run via
    // `npm run test:e2e`) - excluding it here keeps `npm test` fast and
    // avoids Vitest trying to execute Playwright's API as if it were its own.
    exclude: ["e2e/**", "node_modules/**"],
  },
});
