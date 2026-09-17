import { expect, test } from "@playwright/test";

/**
 * Full golden-path flow against the real ml-service: ingest -> view ->
 * generate claims -> publish -> view the public read-only page. Requires
 * the ml-service (and Postgres) to already be running - see
 * playwright.config.ts.
 *
 * Ingestion timeout is env-dependent: locally (Docker/dev machine) a full
 * paper embeds in ~10-15s. Against the deployed Render free tier (0.1
 * vCPU), the same 58-chunk paper measured 4-4.5 minutes when warm, and
 * 7+ minutes after the free tier spun the service down from inactivity
 * (cold-start tax on top of the CPU-bound work) - a real constraint of
 * the free tier, not a bug (see ROADMAP.md; a keep-warm ping mitigates
 * the sleep/cold-start part but not the intrinsic embedding time). Give
 * it real headroom rather than a flaky local-sized timeout when pointed
 * at a remote deployment.
 */
const isRemote = !!process.env.E2E_BASE_URL && !process.env.E2E_BASE_URL.includes("localhost");
const INGEST_TIMEOUT = isRemote ? 10 * 60_000 : 60_000;

test("ingest, generate claims, publish, and view the public page", async ({ page }) => {
  test.setTimeout(isRemote ? 12 * 60_000 : 90_000);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Veritas" })).toBeVisible();

  await page.getByLabel("arXiv ID").fill("1706.03762");
  await page.getByRole("button", { name: "Ingest paper" }).click();

  // Parsing + embedding a real paper takes real time.
  await page.waitForURL(/\/documents\//, { timeout: INGEST_TIMEOUT });
  await expect(page.getByRole("heading", { name: "Attention Is All You Need" })).toBeVisible({
    timeout: 15_000,
  });

  // The playground renders and its slider is interactive.
  await expect(page.getByText(/Playground/)).toBeVisible();

  await page.getByLabel("Query / section topic").fill("How does multi-head attention work?");
  await page.getByRole("button", { name: "Generate claims" }).click();

  const claimsHeading = page.getByRole("heading", { name: /Evidence-grounded claims/ });
  await expect(claimsHeading).toHaveText(/Evidence-grounded claims \([1-9]\d*\)/, {
    timeout: 30_000,
  });

  // At least one grounding badge rendered (supported/partial/unsupported).
  await expect(page.getByText(/supported|partial|unsupported/i).first()).toBeVisible();

  await page.getByRole("button", { name: "Publish" }).click();
  const link = page.getByRole("link", { name: /\/p\// });
  await expect(link).toBeVisible({ timeout: 15_000 });
  const publicUrl = await link.getAttribute("href");
  expect(publicUrl).toMatch(/\/p\/.+/);

  await page.goto(publicUrl!);
  await expect(page.getByText("Published, read-only")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Attention Is All You Need" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download .veritas.json" })).toBeVisible();
});
