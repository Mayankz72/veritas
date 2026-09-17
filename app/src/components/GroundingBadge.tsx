import type { GroundingLabel } from "@/lib/schemas/document";

const STYLES: Record<GroundingLabel, string> = {
  supported: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  partial: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  unsupported: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
};

export function GroundingBadge({
  label,
  score,
}: {
  label: GroundingLabel | null;
  score: number | null;
}) {
  if (label === null) {
    return (
      <span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium bg-black/5 text-black/50 dark:bg-white/10 dark:text-white/50">
        not checked
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${STYLES[label]}`}
      title={score !== null ? `confidence ${(score * 100).toFixed(0)}%` : undefined}
    >
      {label}
      {score !== null ? ` · ${(score * 100).toFixed(0)}%` : ""}
    </span>
  );
}
