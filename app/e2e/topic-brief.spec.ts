import { expect, test } from "@playwright/test";

/**
 * Topic flow: type a topic -> preview papers found on arXiv -> build a brief ->
 * watch progress -> read the per-paper briefs and the cross-paper synthesis.
 *
 * Uses the real arXiv API and (if configured) a real LLM, so it needs network,
 * the ml-service and Postgres running, and can take several minutes the first
 * time. Set E2E_TOPIC_NETWORK=1 to include it; SCREENSHOT_DIR saves screenshots.
 */

const shot = (name: string) =>
  process.env.SCREENSHOT_DIR ? `${process.env.SCREENSHOT_DIR}/${name}.png` : undefined;

test("shows an existing completed brief", async ({ page }) => {
  test.skip(!process.env.E2E_TOPIC_ID, "set E2E_TOPIC_ID to a finished topic");
  await page.goto(`/topics/${process.env.E2E_TOPIC_ID}`);
  await expect(page.getByTestId("synthesis")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("paper-card").first()).toBeVisible();
  const path = shot("topic-complete");
  if (path) await page.screenshot({ path, fullPage: true });
});

test("topic -> papers -> brief -> synthesis", async ({ page }) => {
  test.skip(!process.env.E2E_TOPIC_NETWORK, "set E2E_TOPIC_NETWORK=1 (uses arXiv + LLM, slow)");
  test.setTimeout(20 * 60_000);

  await page.goto("/");
  await page.getByTestId("topic-input").fill("BERT pre-training bidirectional transformers");
  await page.getByRole("button", { name: "Find papers" }).click();

  const candidates = page.getByTestId("candidates");
  await expect(candidates).toBeVisible({ timeout: 120_000 });
  const boxes = candidates.getByRole("checkbox");
  expect(await boxes.count()).toBeGreaterThan(1);
  // Keep the run short: analyse only the top two papers.
  for (let i = 2; i < (await boxes.count()); i++) await boxes.nth(i).uncheck();
  const searchShot = shot("topic-candidates");
  if (searchShot) await page.screenshot({ path: searchShot, fullPage: true });

  await page.getByTestId("build-brief").click();
  // Generous: `next dev` compiles the /topics/[id] route on first visit.
  await expect(page).toHaveURL(/\/topics\//, { timeout: 90_000 });

  // Papers analysed for an earlier topic are reused, so a repeat run can finish
  // before the page loads - the progress panel only exists while work is running.
  if (await page.getByTestId("progress").isVisible()) {
    const progressShot = shot("topic-progress");
    if (progressShot) await page.screenshot({ path: progressShot, fullPage: true });
  }

  await expect(page.getByTestId("synthesis")).toBeVisible({ timeout: 18 * 60_000 });
  await expect(page.getByTestId("paper-card")).toHaveCount(2);
  const doneShot = shot("topic-done");
  if (doneShot) await page.screenshot({ path: doneShot, fullPage: true });
});

test("search marks recent papers as new", async ({ page }) => {
  test.skip(!process.env.E2E_TOPIC_NETWORK, "set E2E_TOPIC_NETWORK=1 (uses arXiv)");
  test.setTimeout(3 * 60_000);

  await page.goto("/");
  await page.getByTestId("topic-input").fill("retrieval-augmented generation");
  await page.getByRole("button", { name: "Find papers" }).click();
  const candidates = page.getByTestId("candidates");
  await expect(candidates).toBeVisible({ timeout: 120_000 });

  await expect(candidates.getByText("new", { exact: true }).first()).toBeVisible();
  const path = shot("topic-search-new");
  if (path) await page.screenshot({ path, fullPage: true });
});

test("a brief can be deleted from the recent list (two-step confirm)", async ({
  page,
  request,
}) => {
  test.skip(!process.env.E2E_TOPIC_NETWORK, "set E2E_TOPIC_NETWORK=1 (needs the ml-service + LLM)");
  test.setTimeout(3 * 60_000);
  const api = "http://localhost:8000";

  // A throwaway topic on a paper that is already analysed, so no download is needed.
  const query = `e2e delete me ${Date.now()}`;
  const created = await request.post(`${api}/topics`, {
    data: {
      query,
      papers: [
        {
          arxiv_id: "1706.03762",
          title: "Attention Is All You Need",
          authors: ["Ashish Vaswani"],
          published: "2017-06-12",
          abstract: "The dominant sequence transduction models are based on complex recurrent networks.",
        },
      ],
    },
  });
  expect(created.ok()).toBeTruthy();
  const { id } = await created.json();

  await page.goto("/");
  const row = page.getByRole("listitem").filter({ hasText: query });
  await expect(row).toBeVisible({ timeout: 30_000 });

  const del = row.getByRole("button", { name: /Delete brief/ });
  await del.click(); // first click only arms it
  await expect(row.getByRole("button", { name: /Confirm delete/ })).toBeVisible();
  await expect(row).toBeVisible();
  await row.getByRole("button", { name: /Confirm delete/ }).click();

  await expect(row).toHaveCount(0);
  expect((await request.get(`${api}/topics/${id}`)).status()).toBe(404);
});
