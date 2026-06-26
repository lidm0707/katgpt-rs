# latent-calculator (LatCal)

Modelless natural-language calculator — **no ML weights, no training, no feature flags, zero dependencies**.

Two modelless engines, fused into one pipeline:

1. **Rule-based** — deterministic lexer → extractor → intent → compute (handles currency, percent, average, total cost, multi-digit).
2. **Neuro-symbolic analytical transformer** — a real transformer forward pass (embed → attention → FFN) with **hand-set weights**. Natural-language operation words get embedded into the op slot, so the network maps NL → operation → result.

> Referenced as `LatCalIx` in Plan 244. "Modelless" = no learned weights anywhere. See [`.plans/245_latent_calculator_modelless.md`](../../.plans/245_latent_calculator_modelless.md).

## Run

```sh
cargo run -p latent-calculator
```

Then type natural-language math. `exit` / `quit` / Ctrl-D to leave.

```
> 2 buy 1
result is 3            (transformer: buy → +)
> double 5
result is 10           (transformer: double → ×2)
> 5 eat 2
result is 3            (transformer: eat → −)
> triple 3
result is 9            (transformer: triple → ×3)
> 3 time each item 20$ total
total is 60$           (rule-based fallback)
> why 2 dog die 1
that doesn't look like a math question
```

## How the fused pipeline works

```
input → analytical transformer →┬─ handles it      → result
                                  └─ declines        → plausibility gate
                                                       ├─ not math   → NotMath
                                                       └─ looks ok   → rule-based engine → result/Unknown
```

- **Transformer** is the authority for what it understands: single-digit operands `0–9` with explicit operators (`+ - ×`, symbols or words) **and** natural-language operation words.
- **Rule-based** is the fallback for richer inputs the transformer can't see (currency `$20`, percent `20%`, average, multi-digit, total cost).
- **Plausibility gate** rejects nonsense (no math anchor + noise words like `why 2 dog die 1`).

## Natural-language operation vocabulary (neuro-symbolic)

These words are embedded into the transformer's op slot — compiled symbolic word-meanings into neural weights:

| Word | Operation |
|---|---|
| `buy` `get` `gain` `receive` | `+` (addition) |
| `eat` `lose` `give` `take` `spend` `drop` | `−` (subtraction) |
| `double` | `×2` (unary → implicit operand) |
| `triple` | `×3` (unary → implicit operand) |

To extend the vocabulary, add to `NL_ADD` / `NL_SUB` / the `double`/`triple` cases in `src/transformer.rs`.

## API

```rust
use latent_calculator::Calculator;

// Fused: transformer-first, rule-based fallback. (This is what the REPL uses.)
Calculator::parse_fused("2 buy 1").unwrap();          // → "result is 3"
Calculator::parse_fused("20% of 50").unwrap();        // → "result is 10"  (rule-based)

// Rule-based only.
Calculator::parse("5 plus 3").unwrap();               // → "sum is 8"

// Raw analytical transformer (None outside its vocabulary).
latent_calculator::transformer::evaluate("double 5"); // → Some(10.0)
```

`ParseError` has two variants: `NotMath` (no math signal) and `Unknown` (couldn't compute).

## Build & test

```sh
cargo test -p latent-calculator        # 31 tests, no flags
cargo clippy -p latent-calculator --all-targets
```

## Project layout

```
src/
  lexer.rs        tokenize + classify (Number, Currency, Quantity, Op, Times, Percent)
  extract.rs      operand lists + N times M vs N times disambiguation
  intent.rs       Computation enum + resolution
  engine.rs       compute + answer formatting (currency side, int-trim)
  plausibility.rs NotMath gate (always-on)
  transformer.rs  neuro-symbolic analytical transformer (NL op words → op slot)
  lib.rs          Calculator API + Answer + ParseError
  main.rs         terminal REPL (uses parse_fused)
tests/
  natural_language.rs   integration tests (spec round-trip)
```
