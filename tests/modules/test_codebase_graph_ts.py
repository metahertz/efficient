from efficient.modules.codebase_graph import _extract_symbols

TS = """\
import { draw } from "./ui";

export function add(a: number, b: number): number {
  return helper(a) + b;
}

const mul = (a: number, b: number): number => a * b;

export class Widget {
  render(): string {
    return draw();
  }
}

interface Shape {
  area(): number;
}

type ID = string;

function helper(n: number): number {
  return n;
}
"""

TSX = """\
export function Button(props: {label: string}) {
  return <button>{props.label}</button>;
}
"""

JS = """\
function greet(name) { return format(name); }
class Box { open() { return unlock(); } }
const shout = (s) => s.toUpperCase();
"""


def _by_name(symbols):
    return {s["symbol"]: s for s in symbols}


def test_typescript_symbol_kinds():
    syms = _by_name(_extract_symbols(TS, "src/app.ts", "r"))
    assert syms["add"]["type"] == "function"
    assert syms["mul"]["type"] == "function"       # arrow const
    assert syms["Widget"]["type"] == "class"
    assert syms["render"]["type"] == "method"
    assert syms["Shape"]["type"] == "interface"
    assert syms["ID"]["type"] == "type"
    assert all(s["language"] == "typescript" for s in syms.values())


def test_typescript_call_edges():
    syms = _by_name(_extract_symbols(TS, "src/app.ts", "r"))
    assert "helper" in syms["add"]["references"]
    assert "draw" in syms["Widget"]["references"] or "draw" in syms["render"]["references"]


def test_tsx_extracts_component():
    syms = _by_name(_extract_symbols(TSX, "src/Button.tsx", "r"))
    assert "Button" in syms
    assert syms["Button"]["language"] == "tsx"


def test_javascript_symbols():
    syms = _by_name(_extract_symbols(JS, "src/box.js", "r"))
    assert syms["greet"]["type"] == "function"
    assert syms["Box"]["type"] == "class"
    assert syms["shout"]["type"] == "function"
    assert "format" in syms["greet"]["references"]


def test_unknown_extension_returns_empty():
    assert _extract_symbols("whatever", "notes.txt", "r") == []
