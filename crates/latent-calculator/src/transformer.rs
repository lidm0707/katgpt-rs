//! Mini analytical transformer — hand-set weights for +, −, × over single
//! digits (Plan 245 Option C).
//!
//! No training, no external deps. Every weight matrix is constructed
//! analytically so the forward pass genuinely computes the result:
//!
//! 1. **Position-aware embed** — operand `a` lands in its own slice,
//!    operand `b` in another, the operator in a third. Roles come from
//!    position (like a position-aware embedding), keeping `a` and `b`
//!    distinguishable for the lookup.
//! 2. **Single-head attention** — the read position (the `=` slot) attends to
//!    the three content positions via hand-set Q/K projections (a fixed
//!    positional pattern), gathering `(a, op, b)` into one vector.
//! 3. **ReLU FFN** — a literal truth table: one hidden unit per `(op, a, b)`
//!    combo, gated by ReLU so exactly the matching unit fires.
//! 4. **Linear readout** — emits the result scalar.
//!
//! Because the construction is exact, the raw output is the integer answer
//! (up to float noise); [`evaluate`] rounds it.

use crate::ArithOp;
use crate::lexer::{Token, lex};

// ── Vocabulary / layout ────────────────────────────────────────
// D_MODEL slice layout:
//   [ 0..10)  operand-a one-hot
//   [10..20)  operand-b one-hot
//   [20..23)  operator one-hot   (Add, Sub, Mul)
//   [23..27)  positional one-hot (positions 0..=3)
const A_DIMS: usize = 10;
const B_SLICE: usize = 10;
const OP_SLICE: usize = 20;
const N_OPS: usize = 3;
const POS_SLICE: usize = 23;
const N_POS: usize = 4;
const CONTENT_DIM: usize = OP_SLICE + N_OPS; // 23 — what the FFN reads
const D_MODEL: usize = POS_SLICE + N_POS; // 27

const POS_OP_A: usize = 0;
const POS_OP: usize = 1;
const POS_OP_B: usize = 2;
const POS_EQ: usize = 3;

/// Attention logits use this temperature so softmax is near-hard; the exact
/// weight `s` is recomputed each forward pass so the result stays exact.
const ATTN_TEMP: f64 = 10.0;

/// FFN gate midpoint: matching combo scores `3s`, nearest rival `2s`, so a
/// threshold of `2.5s` separates them with a full `s` margin.
const GATE_MID_MULT: f64 = 2.5;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpIdx {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy)]
enum Slot {
    OpA(usize),
    Op(OpIdx),
    OpB(usize),
    Eq,
}

/// Run the analytical transformer on `input` and return the integer result.
///
/// Scope: single-digit operands `0..=9` with `+`, `-`, or `×` (also the words
/// `plus / minus / times` and the symbols). Returns `None` for anything
/// outside that vocabulary (multi-digit, division, missing operands).
pub fn evaluate(input: &str) -> Option<f64> {
    let tokens = lex(input);
    let (a, op, b) = parse_seq(&tokens)?;
    Some(forward(a, op, b).round())
}

/// Raw (unrounded) transformer output — exposed for tests proving the forward
/// pass computes the value, not a rounding trick.
pub fn forward(a: usize, op: OpIdx, b: usize) -> f64 {
    let seq = [Slot::OpA(a), Slot::Op(op), Slot::OpB(b), Slot::Eq];

    // 1. embed each position
    let mut emb = [[0.0f64; D_MODEL]; N_POS];
    for (j, s) in seq.iter().enumerate() {
        embed(*s, j, &mut emb[j]);
    }

    // 2. attention from the read position (POS_EQ) over all positions.
    //    Q/K are hand-set 1-dim projections of the positional slice:
    //      q  = x[POS_EQ]
    //      k  = x[pos0] + x[pos1] + x[pos2] - x[pos3]
    //    ⇒ logits = ATTN_TEMP * [1, 1, 1, -1] at the query position.
    let q = emb[POS_EQ][POS_SLICE + POS_EQ];
    let mut logits = [0.0f64; N_POS];
    for j in 0..N_POS {
        let k = emb[j][POS_SLICE + POS_OP_A]
            + emb[j][POS_SLICE + POS_OP]
            + emb[j][POS_SLICE + POS_OP_B]
            - emb[j][POS_SLICE + POS_EQ];
        logits[j] = ATTN_TEMP * q * k;
    }
    let w = softmax(&logits);
    // The three content positions share an equal weight `s`; the self-slot is
    // near-zero and carries no content anyway.
    let s = w[POS_OP_A];

    // attention output at the read position (content dims only)
    let mut ao = [0.0f64; CONTENT_DIM];
    for d in 0..CONTENT_DIM {
        for j in 0..N_POS {
            ao[d] += w[j] * emb[j][d];
        }
    }

    // 3 + 4. FFN truth table: one ReLU unit per (op, a, b). Only the unit
    //    matching the actual combo clears the `2.5s` gate; its readout emits
    //    the result. Scaling by `1/(0.5s)` undoes the gate output (`0.5s`).
    let bias = -GATE_MID_MULT * s;
    let readout_scale = 1.0 / (0.5 * s);
    let mut out = 0.0;
    for oi in 0..N_OPS {
        for aj in 0..A_DIMS {
            for bl in 0..A_DIMS {
                let pre = ao[aj] + ao[B_SLICE + bl] + ao[OP_SLICE + oi] + bias;
                let h = pre.max(0.0);
                out += readout_scale * result_of(oi, aj, bl) * h;
            }
        }
    }
    out
}

fn embed(slot: Slot, pos: usize, out: &mut [f64; D_MODEL]) {
    match slot {
        Slot::OpA(a) => out[a] = 1.0,
        Slot::Op(op) => out[OP_SLICE + op as usize] = 1.0,
        Slot::OpB(b) => out[B_SLICE + b] = 1.0,
        Slot::Eq => {}
    }
    out[POS_SLICE + pos] = 1.0;
}

fn result_of(op: usize, a: usize, b: usize) -> f64 {
    match op {
        0 => (a + b) as f64,
        1 => a as f64 - b as f64,
        _ => (a * b) as f64,
    }
}

fn softmax(logits: &[f64; N_POS]) -> [f64; N_POS] {
    let m = logits.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
    let mut e = [0.0f64; N_POS];
    let mut sum = 0.0;
    for i in 0..N_POS {
        e[i] = (logits[i] - m).exp();
        sum += e[i];
    }
    let inv = 1.0 / sum;
    for v in e.iter_mut() {
        *v *= inv;
    }
    e
}

/// Reduce tokens to `(operand_a, op_idx, operand_b)` for single digits.
fn parse_seq(tokens: &[Token<'_>]) -> Option<(usize, OpIdx, usize)> {
    let mut nums: Vec<f64> = Vec::new();
    let mut op: Option<OpIdx> = None;
    for t in tokens {
        match t {
            Token::Number(n) | Token::Quantity(n) => nums.push(*n),
            Token::Op(ArithOp::Add) => op = op.or(Some(OpIdx::Add)),
            Token::Op(ArithOp::Sub) => op = op.or(Some(OpIdx::Sub)),
            Token::Op(ArithOp::Mul) | Token::Times => op = op.or(Some(OpIdx::Mul)),
            Token::Op(ArithOp::Div) => return None,
            _ => {}
        }
    }
    let op = op?;
    if nums.len() != 2 {
        return None;
    }
    Some((as_digit(nums[0])?, op, as_digit(nums[1])?))
}

fn as_digit(n: f64) -> Option<usize> {
    if n.fract() == 0.0 && (0.0..=9.0).contains(&n) {
        Some(n as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        assert_eq!(evaluate("3 + 5"), Some(8.0));
        assert_eq!(evaluate("7 plus 2"), Some(9.0));
        assert_eq!(evaluate("0 + 0"), Some(0.0));
    }

    #[test]
    fn sub() {
        assert_eq!(evaluate("9 - 4"), Some(5.0));
        assert_eq!(evaluate("2 minus 9"), Some(-7.0));
    }

    #[test]
    fn mul() {
        assert_eq!(evaluate("9 times 9"), Some(81.0));
        assert_eq!(evaluate("6 * 7"), Some(42.0));
        assert_eq!(evaluate("0 * 9"), Some(0.0));
    }

    #[test]
    fn forward_is_exact_within_float_noise() {
        // Proves the transformer computes the value, not a rounding artefact.
        for a in 0..=9 {
            for b in 0..=9 {
                let got = forward(a, OpIdx::Mul, b);
                assert!((got - (a * b) as f64).abs() < 1e-6, "{a}*{b} => {got}");
            }
        }
    }

    #[test]
    fn rejects_out_of_vocab() {
        assert_eq!(evaluate("12 + 3"), None); // multi-digit
        assert_eq!(evaluate("8 / 2"), None); // division unsupported
        assert_eq!(evaluate("5 plus"), None); // missing operand
    }
}
