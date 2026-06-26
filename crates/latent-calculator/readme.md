# latent-calculator (LatCal)

Modelless natural-language calculator — **no ML weights, no training, no feature flags, zero dependencies**.

A neuro-symbolic latent-space engine: a tokenizer feeds a hand-set forward pass that understands natural-language math, then decodes the result symbolically.

> Referenced as `LatCalIx` in Plan 244. "Modelless" = no learned weights anywhere. See [`.plans/245_latent_calculator_modelless.md`](../../.plans/245_latent_calculator_modelless.md).

## Architecture — 3 files

```
src/
  tokenizer.rs   lexical classification (numbers, currency, percent, ops, count-units)
  transformer.rs neuro-symbolic latent-space engine (the brain)
  main.rs        terminal REPL
  lib.rs         thin re-exports
```

### The latent-space pipeline (`transformer.rs`)

```
tokens → embed → attend → Latent → decode → Answer
         (gather operand slots)  (read operation+operands)  (symbolic compute)
```

- **`embed`** — attends to each token's kind/value, gathers latent operand slots (quantities, prices, numbers, percents) + flags.
- **`attend`** — reads the operation + operands out of the latent state into a `Latent` computation. NL operation words are compiled into the op slot here.
- **`decode`** — symbolic arithmetic on the `Latent` (handles arbitrary-precision numbers, currency, percent).
- **Plausibility gate** — no math anchor + noise → `NotMath` (rejects nonsense like `why 2 dog die 1`).

This split is what makes it neuro-symbolic: neural-style understanding (`embed`+`attend`) selects the operation and operands; `decode` does the arithmetic.

## Run

```sh
cargo run -p latent-calculator
```

```
> 3 time each item 20$ total
total is 60$            (TotalCost)
> 5 plus 3
sum is 8                (arithmetic)
> 10$ discount 2%
result is 9.8$          (percent-price)
> 2 buy 1
sum is 3                (NL word → +)
> double 15
product is 30           (NL word → ×2, any magnitude)
> why 2 dog die 1
that doesn't look like a math question
```

## Natural-language operation vocabulary

Compiled into the transformer's op slot (works for any operand magnitude):

| Word | Operation |
|---|---|
| `buy` `get` `gain` `receive` | `+` |
| `eat` `lose` `give` `take` `spend` `drop` | `−` |
| `double` | `×2` |
| `triple` | `×3` |
| `discount` `off` `sale` `save` | price × (1 − pct/100) |
| `tax` `tip` `vat` | price × (1 + pct/100) |

Plus structural words: `total/price/cost/sum` (total cost), `average/avg/mean`, `of` (percent-of), `and/plus` (sum).

## API

```rust
use latent_calculator::Calculator;

Calculator::parse("2 buy 1").unwrap();          // → "sum is 3"
Calculator::parse("10$ discount 2%").unwrap();  // → "result is 9.8$"
```

`ParseError`: `NotMath` (no math signal) or `Unknown` (couldn't compute).

## Build & test

```sh
cargo test -p latent-calculator     # 17 tests, no flags
cargo clippy -p latent-calculator --all-targets
```
