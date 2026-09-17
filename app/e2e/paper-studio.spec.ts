import { expect, test } from "@playwright/test";

/**
 * Full golden-path flow against the real ml-service: ingest -> view ->
 * generate claims -> publish -> view the public read-only page. Requires
 * the ml-service (and Postgres) to already be running - see
 * playwright.config.ts.
 */
test("ingest, generate claims, publish, and view the public page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Veritas" })).toBeVisible();

  await page.getByLabel("arXiv ID").fill("1706.03762");
  await page.getByRole("button", { name: "Ingest paper" }).click();

  // Parsing + embedding a real paper takes real time.
  await page.waitForURL(/\/documents\//, { timeout: 60_000 });
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
