/**
 * Restricted math expression parser/evaluator for interactive formula
 * playgrounds. Deliberately does NOT use eval()/new Function() - it's a
 * hand-rolled recursive-descent parser producing an AST, then a tree-walking
 * evaluator over a fixed allowlist of functions. Untrusted formula strings
 * (paper-derived, or user-edited later) can never execute arbitrary code.
 */

export type MathNode =
  | { type: "num"; value: number }
  | { type: "var"; name: string }
  | { type: "unary"; op: "-"; operand: MathNode }
  | { type: "binop"; op: "+" | "-" | "*" | "/" | "^"; left: MathNode; right: MathNode }
  | { type: "call"; name: string; args: MathNode[] };

export class MathParseError extends Error {}
export class MathEvalError extends Error {}

// A Map (not a plain object) so lookups can never resolve through the
// prototype chain - a plain-object allowlist would let "constructor(1)"
// silently call Object.prototype.constructor instead of being rejected.
const FUNCTIONS = new Map<string, (...args: number[]) => number>([
  ["sqrt", Math.sqrt],
  ["sin", Math.sin],
  ["cos", Math.cos],
  ["tan", Math.tan],
  ["exp", Math.exp],
  ["ln", Math.log],
  ["log", Math.log10],
  ["abs", Math.abs],
  ["max", (...args) => Math.max(...args)],
  ["min", (...args) => Math.min(...args)],
]);

type Token =
  | { type: "number"; value: number }
  | { type: "ident"; value: string }
  | { type: "op"; value: "+" | "-" | "*" | "/" | "^" }
  | { type: "lparen" }
  | { type: "rparen" }
  | { type: "comma" };

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (/[0-9.]/.test(ch)) {
      let j = i;
      while (j < source.length && /[0-9.]/.test(source[j])) j++;
      const raw = source.slice(i, j);
      const value = Number(raw);
      if (Number.isNaN(value)) throw new MathParseError(`Invalid number literal "${raw}"`);
      tokens.push({ type: "number", value });
      i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(ch)) {
      let j = i;
      while (j < source.length && /[a-zA-Z0-9_]/.test(source[j])) j++;
      tokens.push({ type: "ident", value: source.slice(i, j) });
      i = j;
      continue;
    }
    if ("+-*/^".includes(ch)) {
      tokens.push({ type: "op", value: ch as "+" | "-" | "*" | "/" | "^" });
      i++;
      continue;
    }
    if (ch === "(") {
      tokens.push({ type: "lparen" });
      i++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ type: "rparen" });
      i++;
      continue;
    }
    if (ch === ",") {
      tokens.push({ type: "comma" });
      i++;
      continue;
    }
    throw new MathParseError(`Unexpected character "${ch}" at position ${i}`);
  }
  return tokens;
}

class Parser {
  private pos = 0;
  constructor(private tokens: Token[]) {}

  private peek(): Token | undefined {
    return this.tokens[this.pos];
  }

  private next(): Token {
    const t = this.tokens[this.pos];
    if (!t) throw new MathParseError("Unexpected end of expression");
    this.pos++;
    return t;
  }

  parse(): MathNode {
    const node = this.parseExpr();
    if (this.pos !== this.tokens.length) {
      throw new MathParseError(`Unexpected trailing token at position ${this.pos}`);
    }
    return node;
  }

  private peekOp(): (Token & { type: "op" }) | undefined {
    const t = this.peek();
    return t?.type === "op" ? t : undefined;
  }

  private parseExpr(): MathNode {
    let node = this.parseTerm();
    let op = this.peekOp();
    while (op && (op.value === "+" || op.value === "-")) {
      this.next();
      node = { type: "binop", op: op.value, left: node, right: this.parseTerm() };
      op = this.peekOp();
    }
    return node;
  }

  private parseTerm(): MathNode {
    let node = this.parsePower();
    let op = this.peekOp();
    while (op && (op.value === "*" || op.value === "/")) {
      this.next();
      node = { type: "binop", op: op.value, left: node, right: this.parsePower() };
      op = this.peekOp();
    }
    return node;
  }

  private parsePower(): MathNode {
    const base = this.parseUnary();
    const op = this.peekOp();
    if (op && op.value === "^") {
      this.next();
      const exponent = this.parsePower(); // right-associative
      return { type: "binop", op: "^", left: base, right: exponent };
    }
    return base;
  }

  private parseUnary(): MathNode {
    const op = this.peekOp();
    if (op && op.value === "-") {
      this.next();
      return { type: "unary", op: "-", operand: this.parseUnary() };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): MathNode {
    const token = this.next();
    if (token.type === "number") return { type: "num", value: token.value };
    if (token.type === "lparen") {
      const node = this.parseExpr();
      if (this.peek()?.type !== "rparen") throw new MathParseError("Expected closing parenthesis");
      this.next();
      return node;
    }
    if (token.type === "ident") {
      if (this.peek()?.type === "lparen") {
        this.next();
        const args: MathNode[] = [];
        if (this.peek()?.type !== "rparen") {
          args.push(this.parseExpr());
          while (this.peek()?.type === "comma") {
            this.next();
            args.push(this.parseExpr());
          }
        }
        if (this.peek()?.type !== "rparen") throw new MathParseError("Expected closing parenthesis");
        this.next();
        return { type: "call", name: token.value, args };
      }
      return { type: "var", name: token.value };
    }
    throw new MathParseError("Expected a number, variable, or parenthesized expression");
  }
}

export function parseExpression(source: string): MathNode {
  return new Parser(tokenize(source)).parse();
}

export function evaluate(node: MathNode, scope: Record<string, number> = {}): number {
  switch (node.type) {
    case "num":
      return node.value;
    case "var": {
      if (!Object.prototype.hasOwnProperty.call(scope, node.name)) {
        throw new MathEvalError(`Unknown variable "${node.name}"`);
      }
      return scope[node.name];
    }
    case "unary":
      return -evaluate(node.operand, scope);
    case "binop": {
      const left = evaluate(node.left, scope);
      const right = evaluate(node.right, scope);
      switch (node.op) {
        case "+":
          return left + right;
        case "-":
          return left - right;
        case "*":
          return left * right;
        case "/":
          return left / right;
        case "^":
          return Math.pow(left, right);
      }
      break;
    }
    case "call": {
      const fn = FUNCTIONS.get(node.name);
      if (!fn) throw new MathEvalError(`Unknown function "${node.name}"`);
      return fn(...node.args.map((arg) => evaluate(arg, scope)));
    }
  }
  throw new MathEvalError("Unreachable");
}

export function evaluateExpression(source: string, scope: Record<string, number> = {}): number {
  return evaluate(parseExpression(source), scope);
}
