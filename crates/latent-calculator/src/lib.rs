//! LatCal — modelless natural-language calculator.
//!
//! No neural model, no weights: a deterministic lexer → extractor → intent →
//! compute pipeline. "Modelless" in the same sense as `questbench` and the
//! `*_modelless` plans elsewhere in this workspace (no learned weights).

pub mod engine;
pub mod extract;
pub mod intent;
pub mod lexer;
#[cfg(feature = "modelless")]
pub mod underspec;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParseError {
    Unknown,
    /// Modelless routing (Plan 245): input relevance entropy exceeded the
    /// QuestBench `plan_new_threshold`. Carries the normalized score in `[0, 1]`.
    #[cfg(feature = "modelless")]
    Underspecified {
        score: f32,
    },
}

/// Entry point: `Calculator::parse("...")` → `Answer`.
pub struct Calculator;

impl Calculator {
    pub fn parse(input: &str) -> Result<Answer, ParseError> {
        let tokens = lexer::lex(input);
        let operands = extract::extract(&tokens);

        // Modelless pre-flight (Plan 245): route genuinely ambiguous inputs to
        // a typed clarification instead of a silent Unknown. Well-specified
        // inputs fall through to deterministic intent resolution unchanged.
        #[cfg(feature = "modelless")]
        if underspec::needs_clarification(&operands, &underspec_default_config()) {
            return Err(ParseError::Underspecified {
                score: underspec::score(&operands),
            });
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

    /// Normalized-entropy underspecification score in `[0, 1]` for the input.
    /// Only available with the `modelless` feature (Plan 245).
    #[cfg(feature = "modelless")]
    pub fn underspec_score(input: &str) -> f32 {
        let tokens = lexer::lex(input);
        let operands = extract::extract(&tokens);
        underspec::score(&operands)
    }
}

#[cfg(feature = "modelless")]
fn underspec_default_config() -> katgpt_core::UnderspecConfig {
    katgpt_core::UnderspecConfig::default()
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
}
