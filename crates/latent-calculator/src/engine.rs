//! Compute a resolved Computation and format the natural-language answer.

use crate::intent::{Computation, PercentDir};
use crate::{Currency, CurrencySide};

pub fn compute(c: &Computation) -> Option<(f64, &'static str, Option<Currency>, CurrencySide)> {
    match c {
        Computation::TotalCost {
            items,
            currency,
            side,
        } => {
            let total: f64 = items.iter().map(|(q, p)| q * p).sum();
            Some((total, "total", Some(*currency), *side))
        }
        Computation::Arith {
            op,
            values,
            currency,
            side,
        } => {
            let v = apply(op, values)?;
            let label = label_of(*op);
            Some((v, label, *currency, *side))
        }
        Computation::Average { values } => {
            if values.is_empty() {
                return None;
            }
            let sum: f64 = values.iter().sum();
            Some((
                sum / values.len() as f64,
                "average",
                None,
                CurrencySide::Suffix,
            ))
        }
        Computation::PercentOf { rate, base } => {
            Some((rate / 100.0 * base, "result", None, CurrencySide::Suffix))
        }
        Computation::PercentPrice {
            price,
            percent,
            dir,
            currency,
            side,
        } => {
            let factor = match dir {
                PercentDir::Discount => 1.0 - percent / 100.0,
                PercentDir::Tax => 1.0 + percent / 100.0,
            };
            Some((price * factor, "result", *currency, *side))
        }
        Computation::Single {
            value,
            currency,
            side,
        } => Some((*value, "result", *currency, *side)),
        Computation::Unknown => None,
    }
}

fn apply(op: &crate::ArithOp, values: &[f64]) -> Option<f64> {
    let (first, rest) = values.split_first()?;
    Some(match op {
        crate::ArithOp::Add => rest.iter().fold(*first, |a, b| a + b),
        crate::ArithOp::Sub => rest.iter().fold(*first, |a, b| a - b),
        crate::ArithOp::Mul => rest.iter().fold(*first, |a, b| a * b),
        crate::ArithOp::Div => rest.iter().fold(*first, |a, b| a / b),
    })
}

fn label_of(op: crate::ArithOp) -> &'static str {
    match op {
        crate::ArithOp::Add => "sum",
        crate::ArithOp::Sub => "difference",
        crate::ArithOp::Mul => "product",
        crate::ArithOp::Div => "quotient",
    }
}

pub fn fmt_num(v: f64) -> String {
    // Snap float noise: round to 9 decimals, so 55.00000000000001 → 55.0.
    let snapped = (v * 1e9).round() / 1e9;
    if snapped.fract() == 0.0 && snapped.abs() < 1e15 {
        format!("{}", snapped as i64)
    } else {
        format!("{snapped}")
    }
}
