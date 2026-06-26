# Plan 245: Latent Calculator (LatCal) — Modelless Natural-Language Math

**Status:** ✅ Rule-based crate + modelless routing (katgpt-core `questbench`) + mini analytical transformer all done. Percepta transformer-VM path still BLOCKED.
**Date:** 2026-06-26
**Crate:** `latent-calculator` (pure Rust; default zero-dep, opt-in `modelless` and `transformer` features)
**Concept:** "LatCal" — referenced as `LatCalIx` in Plan 244. Modelless = no ML weights, pure rule-based NLU. The `modelless` feature adds QuestBench underspecification routing from `katgpt-core` (Plan 110 / Research 008); the `transformer` feature adds a hand-built analytical transformer (no training) for +, −, × over single digits.

## Goal

Parse natural human-language commands into arithmetic and compute the answer.
No neural model — deterministic, dependency-free, zero-copy lexing.

**Example (spec):**
```
> I buy persona5 3time each item 20$ in what is price total
total is 60$
```

## Why "modelless"

- No weights, no inference runtime, no GPU.
- Rule-based lexer + operand extraction + intent detection + compute.
- Fast, deterministic, auditable. Fits the LatCal "valid index" concept (Plan 244).

## Design

Pipeline (each step a small module):
1. `lexer`  — tokenize NL into typed tokens. Words borrow `&str` (zero-copy).
2. `extract` — resolve tokens into operands: quantities, prices, plain numbers.
3. `intent` — detect operation (TotalCost, Sum, Difference, Product, Quotient, Average, Percent).
4. `engine` — compute + format natural-language answer.

Key heuristics:
- Currency-attached number (`20$`, `$20`) → **price**.
- Count-unit attached number (`3time`, `3x`, `3 times`) → **quantity**.
- Embedded number in a word (`persona5`) → ignored (no leading digits).
- `N times M` (times between two numbers) → **multiply**; `N times` not followed by number → **quantity**.
- qty + price (no `+ - × ÷` operator) → **TotalCost** = qty × price.
- Operators (`plus/minus/times/divided`) on plain numbers → arithmetic.

Preserve currency side from input (`20$` suffix → `60$` suffix).

## Tasks
- [x] 0. Create crate skeleton + workspace member + Cargo.toml
- [x] 1. `lexer.rs` — tokenize + classify (Number, Currency, Quantity, Word, Times, Op, Percent)
- [x] 2. `extract.rs` — operand lists + `N times M` vs `N times` disambiguation
- [x] 3. `intent.rs` — Intent enum + resolution
- [x] 4. `engine.rs` — compute + answer formatting (currency side, int-trim)
- [x] 5. `lib.rs` — public `Calculator` API + `Answer`
- [x] 6. `main.rs` — terminal REPL (read line → answer)
- [x] 7. Integration test: user's spec example → `total is 60$`
- [x] 8. `cargo check && cargo clippy -p latent-calculator` clean
- [x] 9. **Modelless routing from katgpt-core (Plan 245 pick D):** opt-in `modelless` feature → `katgpt-core/questbench`. New `src/underspec.rs` builds a fixed-vocabulary relevance distribution over the 5 computation kinds from `Operands`, then reuses `katgpt_core::underspecification_score` (normalized entropy) + `QuestBenchDecision` + `tier_from_score`. `Calculator::parse` routes genuinely ambiguous inputs (score > `plan_new_threshold`) to a typed `ParseError::Underspecified { score }`; well-specified inputs compute unchanged. Default crate stays zero-dependency.
- [x] 10. **Mini analytical transformer (Plan 245 Option C):** opt-in zero-dependency `transformer` feature → new `src/transformer.rs`. A real hand-weighted forward pass (position-aware embed → single-head attention via hand-set Q/K → ReLU FFN truth table → linear readout) computes `+`, `−`, `×` exactly over single-digit operands. Weights are analytic (no training): one ReLU unit per `(op, a, b)` combo, gated at `2.5s`. `evaluate()`/`forward()` + `Calculator::parse_transformer`. 9 → 9 exhaustive check passes within 1e-6.

## Notes
- Pure std, no external crates (lean, zero-copy).
- Enums for all variants (Currency, QtyUnit, ArithOp, Intent, CurrencySide) — no hardcoded strings.
- `pub mod` style; consume via `crate::` / `super::`.
- No `#[allow(dead_code)]`; unused → `todo!()` (none expected in v1).

## Investigation: "transformer modelless" (percepta / katgpt-core)

User wants the calculator to use a *modelless transformer* (analytically-constructed
weights, zero training). Findings (verified in this checkout):

- **Percepta** (`examples/percepta_phase0.rs`) IS the repo's only real modelless
  transformer: `Runner::build(None)` constructs weights via MILP (Futamura-style
  projection of a WASM interpreter). `Runner::specialize(&[ProgramInstruction], _)`
  bakes a specific program into the weights. `Opcode` has `I32Const/I32Add/I32Sub/
  Output` but **no MUL** (multiply lowered to repeated addition).
- **BLOCKER — `percepta_compile` does not build here:**
  - `src/percepta/compile.rs:55` does `include_str!("../../.raw/transformer-vm/.../runtime.h")`
    but `.raw/` is gitignored (`.gitignore:2`) and absent → hard compile error.
  - `src/percepta/evaluator.rs` has 2 borrow-checker errors: E0382 (`dim_order`
    moved then borrowed, ~L137/156) and E0499 (double `&mut self`, L190/319).
- **`katgpt-core` is NOT a transformer** — it's shared types + SIMD kernels.
  `DomainLatent` (Plan 038) is just a `Vec<f32>` embedding blob loaded from a
  `.bin` file, injected mid-layer into a (trained) transformer. It does no NL→math
  and builds no weights. "modelless" in katgpt-core is only a *mode flag*
  (`HydraBudgetConfig.modelless`) + doc comments.
- The plain `src/transformer.rs` (`TransformerWeights::new`) inits RANDOM weights
  (benchmark-only); no trained model exists in the repo, so it can't do NL→math.

## Decision fork (needs user pick) — RESOLVED: picked D
- **A) Fix Percepta + wire it:** reconstruct/ fetch `.raw/.../runtime.h`, fix the
  2 evaluator borrows, then `latent-calculator` lowers NL→`ProgramInstruction` →
  `Runner::specialize` → `Runner::run` → decode answer. Faithful modelless
  transformer; edits shared `katgpt-rs` internals; multiply needs a loop (slow).
- **B) Keep modelless rule-based crate** (DONE, `cargo test` green): genuinely
  modelless (no neural model), computes `3×20=60` from NL today.
- **C) Tiny custom analytically-built transformer** in the new crate (own weights
  by hand for +,-,× over a digit vocabulary) — no percepta, no katgpt-rs dep. ✅ DONE (2026-06-26): `src/transformer.rs`, opt-in `transformer` feature. Real forward pass (embed → attention → ReLU FFN → readout), all weights analytic. Computes +, −, × exactly for single-digit operands; exhaustive 10×10 check within 1e-6. `evaluate` / `forward` / `Calculator::parse_transformer`.
- **D) Use modelless infra from `katgpt-core`** ✅ DONE (2026-06-26): wire the
  `questbench` underspecification scorer into the calculator. `katgpt-core` is
  not a transformer, but its `questbench` module IS a real modelless component
  (normalized-entropy underspecification + QuestBench decision thresholds,
  Plan 110 / Research 008). Implemented behind opt-in `modelless` feature so the
  default crate keeps its zero-dependency guarantee. Files: `Cargo.toml`, new
  `src/underspec.rs`, `src/lib.rs` (`ParseError::Underspecified`, routing in
  `Calculator::parse`, `Calculator::underspec_score`). Verified: 19 tests green
  with `--features modelless`, 14 green default, clippy clean on the crate.
