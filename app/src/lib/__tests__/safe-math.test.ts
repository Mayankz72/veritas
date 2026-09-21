import { describe, expect, it } from "vitest";
import { MathEvalError, MathParseError, evaluateExpression } from "../safe-math";

describe("evaluateExpression", () => {
  it("evaluates basic arithmetic with correct precedence", () => {
    expect(evaluateExpression("2 + 3 * 4")).toBe(14);
    expect(evaluateExpression("(2 + 3) * 4")).toBe(20);
    expect(evaluateExpression("10 / 2 - 1")).toBe(4);
  });

  it("supports exponentiation as right-associative", () => {
    expect(evaluateExpression("2 ^ 3")).toBe(8);
    expect(evaluateExpression("2 ^ 3 ^ 2")).toBe(2 ** (3 ** 2));
  });

  it("supports unary minus", () => {
    expect(evaluateExpression("-5 + 3")).toBe(-2);
    expect(evaluateExpression("3 - -2")).toBe(5);
  });

  it("substitutes variables from scope", () => {
    expect(evaluateExpression("qk / sqrt(d_k)", { qk: 64, d_k: 64 })).toBe(8);
  });

  it("supports the allowlisted functions", () => {
    expect(evaluateExpression("sqrt(64)")).toBe(8);
    expect(evaluateExpression("max(1, 5, 3)")).toBe(5);
    expect(evaluateExpression("min(1, 5, 3)")).toBe(1);
    expect(evaluateExpression("abs(-7)")).toBe(7);
  });

  it("computes the scaled dot-product attention denominator from the paper", () => {
    // Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V - this checks the
    // scaling term in isolation, driven by a slider over d_k in the UI.
    const dotProduct = 512;
    expect(evaluateExpression("dot / sqrt(d_k)", { dot: dotProduct, d_k: 64 })).toBeCloseTo(64, 5);
  });

  it("throws MathParseError on invalid syntax, never silently returns NaN", () => {
    expect(() => evaluateExpression("2 + ")).toThrow(MathParseError);
    expect(() => evaluateExpression("(2 + 3")).toThrow(MathParseError);
    expect(() => evaluateExpression("2 3")).toThrow(MathParseError);
  });

  it("throws MathEvalError for unknown variables and functions", () => {
    expect(() => evaluateExpression("x + 1")).toThrow(MathEvalError);
    expect(() => evaluateExpression("nope(1)")).toThrow(MathEvalError);
  });

  it("never executes arbitrary JS - constructor/prototype access is just an unknown identifier", () => {
    expect(() => evaluateExpression("constructor(1)")).toThrow(MathEvalError);
    expect(() => evaluateExpression("__proto__")).toThrow(MathEvalError);
  });
});
