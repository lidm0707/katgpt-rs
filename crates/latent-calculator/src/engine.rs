//! Compute a resolved Computation and format the natural-language answer.

use crate::intent::Computation;
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
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v}");
        s
    }
}
