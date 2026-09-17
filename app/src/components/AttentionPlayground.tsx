"use client";

import { useMemo, useState } from "react";
import { Formula } from "./Formula";
import { evaluateExpression } from "@/lib/safe-math";

const SCALING_EXPRESSION = "dot / sqrt(d_k)";

/**
 * Interactive playground for the paper's "Scaled Dot-Product Attention"
 * (section 3.2.1): as d_k grows, raw dot products grow in magnitude and can
 * push softmax into a near-zero-gradient region - scaling by sqrt(d_k)
 * counteracts that. Sliders recompute via the sandboxed evaluator in
 * lib/safe-math.ts (no eval/new Function).
 */
export function AttentionPlayground() {
  const [dotProduct, setDotProduct] = useState(64);
  const [dK, setDK] = useState(64);

  const scaled = useMemo(
    () => evaluateExpression(SCALING_EXPRESSION, { dot: dotProduct, d_k: dK }),
    [dotProduct, dK],
  );

  // A softmax with inputs beyond roughly +/-10 has already saturated into a
  // near-one-hot distribution - illustrative threshold, not from the paper.
  const saturationRatio = Math.min(Math.abs(scaled) / 10, 1);

  return (
    <div className="rounded-lg border border-black/10 dark:border-white/15 p-5 space-y-5">
      <div>
        <h3 className="font-semibold text-sm uppercase tracking-wide text-black/60 dark:text-white/60">
          Playground &middot; 3.2.1 Scaled Dot-Product Attention
        </h3>
        <div className="mt-2 text-lg">
          <Formula block tex="\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />
        </div>
      </div>

      <p className="text-sm text-black/70 dark:text-white/70">
        The paper scales the raw query-key dot product by{" "}
        <Formula tex="1/\sqrt{d_k}" /> because for large <Formula tex="d_k" />, dot
        products grow large in magnitude and push softmax into regions with
        extremely small gradients. Drag the sliders to see the scaled value
        shrink back toward a well-behaved range as <Formula tex="d_k" /> grows.
      </p>

      <div className="space-y-4">
        <label className="block text-sm">
          <span className="flex justify-between">
            <span>Raw dot product (q &middot; k)</span>
            <span className="font-mono">{dotProduct}</span>
          </span>
          <input
            type="range"
            min={-256}
            max={256}
            value={dotProduct}
            onChange={(e) => setDotProduct(Number(e.target.value))}
            className="w-full"
          />
        </label>

        <label className="block text-sm">
          <span className="flex justify-between">
            <span>
              d_k (key dimension per head; paper uses 64 = d_model/h = 512/8)
            </span>
            <span className="font-mono">{dK}</span>
          </span>
          <input
            type="range"
            min={1}
            max={512}
            value={dK}
            onChange={(e) => setDK(Number(e.target.value))}
            className="w-full"
          />
        </label>
      </div>

      <div className="flex items-center gap-4">
        <div className="font-mono text-lg">
          {dotProduct} / &radic;{dK} = <strong>{scaled.toFixed(3)}</strong>
        </div>
        <div className="flex-1 h-2 rounded bg-black/10 dark:bg-white/10 overflow-hidden">
          <div
            className="h-full bg-amber-500 transition-all"
            style={{ width: `${saturationRatio * 100}%` }}
          />
        </div>
        <span className="text-xs text-black/50 dark:text-white/50 w-32 text-right">
          {saturationRatio > 0.8 ? "near-saturated softmax" : "well-scaled"}
        </span>
      </div>
    </div>
  );
}
