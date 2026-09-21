import { IngestForm } from "@/components/IngestForm";
import { TopicSearch } from "@/components/TopicSearch";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex flex-1 w-full max-w-3xl flex-col items-center gap-10 py-20 px-6 text-center">
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold tracking-tight">Veritas</h1>
          <p className="max-w-xl text-black/60 dark:text-white/60">
            Enter a research topic. Veritas finds the papers, reads each one, tells you the
            problem, method, key results and why it matters, and shows how the papers fit
            together. Every statement links back to a page and a quote in the paper.
          </p>
        </div>

        <TopicSearch />

        <details className="w-full max-w-md text-left">
          <summary className="cursor-pointer text-sm text-black/50 dark:text-white/50 text-center">
            Or open a single paper by arXiv ID
          </summary>
          <div className="pt-4">
            <IngestForm />
          </div>
        </details>
      </main>
    </div>
  );
}
