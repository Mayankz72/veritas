import { IngestForm } from "@/components/IngestForm";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex flex-1 w-full max-w-3xl flex-col items-center justify-center gap-8 py-32 px-6 text-center">
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold tracking-tight">Veritas</h1>
          <p className="max-w-md text-black/60 dark:text-white/60">
            Turn a research paper into an evidence-grounded workspace. Every
            claim links back to the exact page and quote it rests on,
            verified by a real retrieval + grounding pipeline.
          </p>
        </div>
        <IngestForm />
      </main>
    </div>
  );
}
