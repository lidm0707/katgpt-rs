//! Operand extraction: walk tokens left-to-right and resolve the
//! ambiguous `N times M` (multiply) vs `N times` (quantity) cases.

use crate::lexer::Token;
use crate::{ArithOp, Currency, CurrencySide};

#[derive(Debug, Default)]
pub struct Operands {
    pub quantities: Vec<f64>,
    pub prices: Vec<(f64, Currency)>,
    pub numbers: Vec<f64>,
    pub ops: Vec<ArithOp>,
    pub percents: Vec<f64>,
    pub has_percent: bool,
    pub has_of: bool,
    pub has_avg: bool,
    pub has_total: bool,
    pub has_sum: bool,
    pub currency: Option<Currency>,
    pub side: CurrencySide,
}

pub fn extract(tokens: &[Token<'_>]) -> Operands {
    let mut o = Operands::default();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            Token::Number(n) => {
                let next = tokens.get(i + 1);
                if let Some(Token::Times) = next {
                    // "N times M" → multiply; "N times" (no following number) → quantity.
                    if matches!(tokens.get(i + 2), Some(Token::Number(_))) {
                        o.numbers.push(*n);
                        o.ops.push(ArithOp::Mul);
                        i += 2; // consume number + times; next number handled next loop
                        continue;
                    }
                    o.quantities.push(*n);
                    i += 2; // consume number + times
                    continue;
                }
                if let Some(Token::Word(w)) = next
                    && crate::lexer::is_count_unit(w)
                {
                    o.quantities.push(*n);
                    i += 2; // consume number + count-unit word
                    continue;
                }
                o.numbers.push(*n);
            }
            Token::Quantity(q) => o.quantities.push(*q),
            Token::Currency { value, cur, side } => {
                o.prices.push((*value, *cur));
                if o.currency.is_none() {
                    o.currency = Some(*cur);
                    o.side = *side;
                }
            }
            Token::PercentValue(v) => {
                o.percents.push(*v);
                o.has_percent = true;
            }
            Token::Percent => o.has_percent = true,
            Token::Op(op) => o.ops.push(*op),
            Token::Word(w) => classify_word(w, &mut o),
            Token::Times => {
                // standalone times not preceded by a number: treat as multiply op
                o.ops.push(ArithOp::Mul);
            }
        }
        i += 1;
    }
    o
}

fn classify_word(w: &str, o: &mut Operands) {
    match w.to_ascii_lowercase().as_str() {
        "of" => o.has_of = true,
        "by" => {}
        "average" | "avg" | "mean" => o.has_avg = true,
        "total" | "altogether" | "price" | "cost" | "sum" | "add" => o.has_total = true,
        "plus" | "and" => o.has_sum = true,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    #[test]
    fn spec_extracts_qty_and_price() {
        let t = lex("I buy persona5 3time each item 20$ in what is price total");
        let o = extract(&t);
        assert_eq!(o.quantities, vec![3.0]);
        assert_eq!(o.prices, vec![(20.0, Currency::Dollar)]);
        assert!(o.has_total);
        assert_eq!(o.side, CurrencySide::Suffix);
    }

    #[test]
    fn times_between_numbers_is_mul() {
        let t = lex("3 times 4");
        let o = extract(&t);
        assert_eq!(o.numbers, vec![3.0, 4.0]);
        assert_eq!(o.ops, vec![ArithOp::Mul]);
        assert!(o.quantities.is_empty());
    }
}
