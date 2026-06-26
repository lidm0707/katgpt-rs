//! Intent resolution: decide what computation the operands imply.

use crate::extract::Operands;
use crate::{ArithOp, Currency, CurrencySide};

#[derive(Debug, Clone, PartialEq)]
pub enum Computation {
    TotalCost {
        items: Vec<(f64, f64)>,
        currency: Currency,
        side: CurrencySide,
    },
    Arith {
        op: ArithOp,
        values: Vec<f64>,
        currency: Option<Currency>,
        side: CurrencySide,
    },
    Average {
        values: Vec<f64>,
    },
    PercentOf {
        rate: f64,
        base: f64,
    },
    Single {
        value: f64,
        currency: Option<Currency>,
        side: CurrencySide,
    },
    Unknown,
}

pub fn resolve(o: &Operands) -> Computation {
    if let Some(c) = try_percent(o) {
        return c;
    }
    if o.has_avg && !o.numbers.is_empty() {
        return Computation::Average {
            values: o.numbers.clone(),
        };
    }
    if !o.quantities.is_empty() && !o.prices.is_empty() && o.ops.is_empty() {
        return total_cost(o);
    }
    if !o.ops.is_empty() {
        return Computation::Arith {
            op: o.ops[0],
            values: value_list(o),
            currency: o.currency,
            side: o.side,
        };
    }
    if (o.has_total || o.has_sum) && value_list(o).len() >= 2 {
        return Computation::Arith {
            op: ArithOp::Add,
            values: value_list(o),
            currency: o.currency,
            side: o.side,
        };
    }
    match value_list(o).as_slice() {
        [v] => Computation::Single {
            value: *v,
            currency: o.currency,
            side: o.side,
        },
        [] if !o.prices.is_empty() => Computation::Single {
            value: o.prices[0].0,
            currency: o.currency,
            side: o.side,
        },
        _ => Computation::Unknown,
    }
}

fn try_percent(o: &Operands) -> Option<Computation> {
    if o.percents.is_empty() || !o.has_of || o.numbers.is_empty() {
        return None;
    }
    Some(Computation::PercentOf {
        rate: o.percents[0],
        base: o.numbers[0],
    })
}

fn total_cost(o: &Operands) -> Computation {
    let items: Vec<(f64, f64)> = match (o.quantities.len(), o.prices.len()) {
        (q, p) if q == p => o
            .quantities
            .iter()
            .copied()
            .zip(o.prices.iter().map(|(p, _)| *p))
            .collect(),
        (1, 1) => vec![(o.quantities[0], o.prices[0].0)],
        _ => o
            .quantities
            .iter()
            .flat_map(|&q| o.prices.iter().map(move |&(p, _)| (q, p)))
            .collect(),
    };
    Computation::TotalCost {
        items,
        currency: o.currency.unwrap_or(Currency::Dollar),
        side: o.side,
    }
}

fn value_list(o: &Operands) -> Vec<f64> {
    let mut v: Vec<f64> = o.numbers.clone();
    if v.is_empty() {
        v.extend(o.prices.iter().map(|(p, _)| *p));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::extract;
    use crate::lexer::lex;

    fn resolve_str(s: &str) -> Computation {
        resolve(&extract(&lex(s)))
    }

    #[test]
    fn spec_is_total_cost() {
        let c = resolve_str("I buy persona5 3time each item 20$ in what is price total");
        assert_eq!(
            c,
            Computation::TotalCost {
                items: vec![(3.0, 20.0)],
                currency: Currency::Dollar,
                side: CurrencySide::Suffix
            }
        );
    }

    #[test]
    fn percent_of() {
        let c = resolve_str("20% of 50");
        assert_eq!(
            c,
            Computation::PercentOf {
                rate: 20.0,
                base: 50.0
            }
        );
    }
}
