//! LatCal — modelless natural-language calculator.
//!
//! No neural model, no weights: a deterministic lexer → extractor → intent →
//! compute pipeline, plus a neuro-symbolic analytical transformer that maps
//! natural-language operation words (`buy`/`eat`/`double`/…) into hand-set
//! transformer weights. "Modelless" = no learned weights anywhere.

pub mod engine;
pub mod extract;
pub mod intent;
pub mod lexer;
pub mod plausibility;
pub mod transformer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Currency {
    Dollar,
    Euro,
    Pound,
    Yen,
}

impl Currency {
    pub fn symbol(self) -> &'static str {
        match self {
            Currency::Dollar => "$",
            Currency::Euro => "€",
            Currency::Pound => "£",
            Currency::Yen => "¥",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurrencySide {
    #[default]
    Suffix,
    Prefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A computed answer ready to be rendered as a natural-language sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub value: f64,
    pub label: &'static str,
    pub currency: Option<Currency>,
    pub side: CurrencySide,
}

impl Answer {
    pub fn to_sentence(&self) -> String {
        let num = engine::fmt_num(self.value);
        match (self.currency, self.side) {
            (Some(cur), CurrencySide::Suffix) => {
                format!("{} is {}{}", self.label, num, cur.symbol())
            }
            (Some(cur), CurrencySide::Prefix) => {
                format!("{} is {}{}", self.label, cur.symbol(), num)
            }
            (None, _) => format!("{} is {}", self.label, num),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    Unknown,
    /// Input does not look like a math command (no strong anchor + noise).
    /// Always-on plausibility gate (Plan 245).
    NotMath,
}

/// Entry point: `Calculator::parse("...")` → `Answer`.
pub struct Calculator;

impl Calculator {
    pub fn parse(input: &str) -> Result<Answer, ParseError> {
        let tokens = lexer::lex(input);
        let operands = extract::extract(&tokens);

        // Always-on plausibility gate (Plan 245): reject inputs that don't look
        // like math (no strong anchor surrounded by noise words).
        if !plausibility::is_plausible_math(&tokens, &operands) {
            return Err(ParseError::NotMath);
        }

        let computation = intent::resolve(&operands);
        match engine::compute(&computation) {
            Some((value, label, currency, side)) => Ok(Answer {
                value,
                label,
                currency,
                side,
            }),
            None => Err(ParseError::Unknown),
        }
    }

    /// Compute via the mini analytical transformer (Plan 245 Option C).
    /// Hand-set weights do `+`, `-`, `×` over single-digit operands, and map
    /// natural-language operation words (`buy`/`eat`/`double`/…) into the op
    /// slot. Returns `None` outside that vocabulary.
    pub fn parse_transformer(input: &str) -> Option<Answer> {
        transformer::evaluate(input).map(|v| Answer {
            value: v,
            label: "result",
            currency: None,
            side: CurrencySide::Suffix,
        })
    }

    /// Fused pipeline (Plan 245): neuro-symbolic transformer first, rule-based
    /// fallback. The analytical transformer maps NL operation words + single-
    /// digit arithmetic; anything it declines (currency, percent, average,
    /// multi-digit) falls through to the rule-based engine with the always-on
    /// plausibility gate.
    pub fn parse_fused(input: &str) -> Result<Answer, ParseError> {
        if let Some(v) = transformer::evaluate(input) {
            return Ok(Answer {
                value: v,
                label: "result",
                currency: None,
                side: CurrencySide::Suffix,
            });
        }

        let tokens = lexer::lex(input);
        let operands = extract::extract(&tokens);
        if !plausibility::is_plausible_math(&tokens, &operands) {
            return Err(ParseError::NotMath);
        }

        let computation = intent::resolve(&operands);
        match engine::compute(&computation) {
            Some((value, label, currency, side)) => Ok(Answer {
                value,
                label,
                currency,
                side,
            }),
            None => Err(ParseError::Unknown),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example() {
        let a = Calculator::parse("I buy persona5 3time each item 20$ in what is price total")
            .expect("spec must parse");
        assert_eq!(a.value, 60.0);
        assert_eq!(a.currency, Some(Currency::Dollar));
        assert_eq!(a.side, CurrencySide::Suffix);
        assert_eq!(a.to_sentence(), "total is 60$");
    }

    #[test]
    fn rejects_non_math_noise() {
        // Always-on plausibility gate: no strong anchor + noise → NotMath.
        assert_eq!(
            Calculator::parse("why 2 dog and die 1"),
            Err(ParseError::NotMath)
        );
        assert_eq!(
            Calculator::parse("the quick brown fox"),
            Err(ParseError::NotMath)
        );
        assert_eq!(Calculator::parse("hello world"), Err(ParseError::NotMath));
    }

    #[test]
    fn terse_pure_math_still_works() {
        // No strong anchor, but no noise → accepted.
        assert_eq!(
            Calculator::parse("5 and 3").unwrap().to_sentence(),
            "sum is 8"
        );
        assert_eq!(
            Calculator::parse("20").unwrap().to_sentence(),
            "result is 20"
        );
    }

    #[test]
    fn fused_transformer_path_single_digit() {
        // Router says well-specified; transformer handles single-digit ×.
        let a = Calculator::parse_fused("9 times 9").expect("fused must parse");
        assert_eq!(a.value, 81.0);
        assert_eq!(a.to_sentence(), "result is 81");
    }

    #[test]
    fn fused_rulebased_fallback_for_percent() {
        // Outside transformer vocab → rule-based fallback preserves semantics.
        let a = Calculator::parse_fused("20% of 50").expect("fused fallback");
        assert_eq!(a.to_sentence(), "result is 10");
    }

    #[test]
    fn fused_router_rejects_ambiguous() {
        // Terse but meaningless (no operator) → rule-based Unknown.
        assert_eq!(Calculator::parse_fused("20 30"), Err(ParseError::Unknown));
    }

    #[test]
    fn fused_spec_falls_back_to_rulebased() {
        // Spec input is well-specified but outside transformer vocab → rule-based.
        let a =
            Calculator::parse_fused("I buy persona5 3time each item 20$ in what is price total")
                .expect("fused spec fallback");
        assert_eq!(a.to_sentence(), "total is 60$");
    }

    #[test]
    fn fused_nl_words_via_transformer() {
        // Neuro-symbolic NL mapping runs in the fused path too.
        assert_eq!(
            Calculator::parse_fused("2 buy 1").unwrap().to_sentence(),
            "result is 3"
        );
        assert_eq!(
            Calculator::parse_fused("double 5").unwrap().to_sentence(),
            "result is 10"
        );
    }
}
