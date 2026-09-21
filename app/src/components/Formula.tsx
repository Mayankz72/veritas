"use client";

import katex from "katex";
import { useMemo } from "react";

export function Formula({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        throwOnError: false,
        displayMode: block,
      }),
    [tex, block],
  );

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
