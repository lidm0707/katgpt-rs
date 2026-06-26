//! Modelless plausibility gate — "is this sentence actually math?"
//!
//! The rule-based engine treats any two numbers joined by `and` as a sum and
//! ignores surrounding words, so nonsense like "why 2 dog and die 1" answers
//! "sum is 3". This gate rejects such inputs before any engine runs.
//!
//! Rule (always-on, zero-dependency):
//! - a sentence is plausibly math if it has a **strong anchor** — an explicit
//!   operator, currency, quantity, percent, or a `total`/`average` keyword; OR
//! - it is terse pure-math with no noise words (e.g. "5 and 3", "20").
//!
//! Natural-language math ("I buy ... 3time ... 20$ ... total") always carries a
//! strong anchor, so it passes; random nouns around two numbers do not.

use crate::extract::Operands;
use crate::lexer::Token;

/// `true` if the token stream looks like a genuine math command.
pub fn is_plausible_math(tokens: &[Token<'_>], o: &Operands) -> bool {
    if has_strong_anchor(tokens, o) {
        return true;
    }
    // No strong anchor: accept only terse inputs with no noise words.
    !has_noise(tokens)
}

fn has_strong_anchor(tokens: &[Token<'_>], o: &Operands) -> bool {
    if o.has_total || o.has_avg || o.has_percent {
        return true;
    }
    tokens.iter().any(|t| match t {
        Token::Op(_)
        | Token::Times
        | Token::Currency { .. }
        | Token::Quantity(_)
        | Token::PercentValue(_)
        | Token::Percent => true,
        // NL operation words (buy/eat/double/…) are math signal too.
        Token::Word(w) => crate::transformer::is_nl_op_word(w),
        _ => false,
    })
}

fn has_noise(tokens: &[Token<'_>]) -> bool {
    tokens
        .iter()
        .any(|t| matches!(t, Token::Word(w) if !is_math_keyword(w)))
}

/// Words the extractor treats as structural/math cues (aligned with
/// `extract::classify_word` + operator words). Anything else is noise.
fn is_math_keyword(w: &str) -> bool {
    [
        "of",
        "by",
        "and",
        "average",
        "avg",
        "mean",
        "total",
        "altogether",
        "price",
        "cost",
        "sum",
    ]
    .iter()
    .any(|k| w.eq_ignore_ascii_case(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::extract;
    use crate::lexer::lex;

    fn check(s: &str) -> bool {
        let t = lex(s);
        is_plausible_math(&t, &extract(&t))
    }

    #[test]
    fn rejects_nonsense_with_noise() {
        assert!(!check("why 2 dog and die 1"));
        assert!(!check("the quick brown fox"));
        assert!(!check("hello world"));
    }

    #[test]
    fn accepts_strong_anchors() {
        assert!(check(
            "I buy persona5 3time each item 20$ in what is price total"
        ));
        assert!(check("3 copies at $20 total"));
        assert!(check("5 plus 3"));
        assert!(check("3 times 4"));
        assert!(check("average of 4 8 and 12"));
        assert!(check("20% of 50"));
        assert!(check("2 buy 1")); // NL operation word = strong anchor
        assert!(check("double 5"));
    }

    #[test]
    fn accepts_terse_pure_math() {
        assert!(check("5 and 3"));
        assert!(check("20"));
        assert!(check("20 30")); // terse → plausible (router handles its ambiguity)
    }
}
