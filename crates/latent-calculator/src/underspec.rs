//! Modelless underspecification routing — reuses katgpt-core's questbench.
//!
//! The calculator parses operands deterministically (no ML weights). This module
//! turns those operands into a relevance distribution over the computation kinds
//! the calculator supports, then feeds that distribution to
//! `katgpt_core::underspecification_score` (normalized entropy) and the
//! QuestBench decision/tier thresholds. High entropy ⇒ the input is ambiguous
//! and should be clarified rather than silently guessed.

use crate::extract::Operands;
use katgpt_core::{MemoryTier, QuestBenchDecision, UnderspecConfig, underspecification_score};

/// One relevance slot per computation kind. Order is internal and fixed so the
/// score stays comparable across inputs.
const INTENT_SLOTS: usize = 5;
const SLOT_TOTAL_COST: usize = 0;
const SLOT_ARITH: usize = 1;
const SLOT_AVERAGE: usize = 2;
const SLOT_PERCENT: usize = 3;
const SLOT_SINGLE: usize = 4;

/// Scale applied to operator evidence so an explicit op dominates other cues.
const OP_EVIDENCE: f32 = 2.0;

/// Relevance weights for each computation kind implied by the operands.
///
/// Slots: `[TotalCost, Arith, Average, Percent, Single]`. Weights are
/// non-negative evidence counts; zero slots are retained so the distribution
/// stays a fixed-length vocabulary (`INTENT_SLOTS`) and entropy stays
/// comparable across inputs.
pub fn relevance(o: &Operands) -> [f32; INTENT_SLOTS] {
    let n_values = value_count(o);
    let mut r = [0.0f32; INTENT_SLOTS];

    r[SLOT_TOTAL_COST] = total_cost_weight(o);
    r[SLOT_ARITH] = arith_weight(o, n_values);
    r[SLOT_AVERAGE] = average_weight(o, n_values);
    r[SLOT_PERCENT] = percent_weight(o);
    r[SLOT_SINGLE] = single_weight(o, n_values);

    r
}

/// Normalized-entropy underspecification score in `[0, 1]`.
/// `0.0` = one dominant intent (well-specified); `1.0` = no evidence at all.
pub fn score(o: &Operands) -> f32 {
    let r = relevance(o);
    underspecification_score(&r)
}

/// QuestBench planning decision for the parsed operands.
pub fn decision(o: &Operands, cfg: &UnderspecConfig) -> QuestBenchDecision {
    QuestBenchDecision::from_score(score(o), cfg)
}

/// Four-tier memory trigger for the parsed operands.
pub fn tier(o: &Operands, cfg: &UnderspecConfig) -> MemoryTier {
    katgpt_core::tier_from_score(score(o), cfg)
}

/// `true` when the input is too ambiguous to compute confidently.
/// Mirrors QuestBench's "a brand-new plan is needed" threshold.
pub fn needs_clarification(o: &Operands, cfg: &UnderspecConfig) -> bool {
    matches!(decision(o, cfg), QuestBenchDecision::PlanNew)
}

fn total_cost_weight(o: &Operands) -> f32 {
    if o.ops.is_empty() && !o.quantities.is_empty() && !o.prices.is_empty() {
        o.quantities.len().min(o.prices.len()) as f32
    } else {
        0.0
    }
}

fn arith_weight(o: &Operands, n_values: usize) -> f32 {
    // Explicit operators are unambiguous — always count them.
    let from_ops = o.ops.len() as f32 * OP_EVIDENCE;
    // Keyword cues (total/sum/and) only count when no stronger intent keyword
    // (average) has claimed its slot; otherwise list conjunctions like "and"
    // would spuriously compete with an explicit average request.
    let from_keywords = if !o.has_avg && (o.has_total || o.has_sum) && n_values >= 2 {
        1.0
    } else {
        0.0
    };
    from_ops + from_keywords
}

fn average_weight(o: &Operands, n_values: usize) -> f32 {
    if o.has_avg && n_values > 0 {
        n_values as f32
    } else {
        0.0
    }
}

fn percent_weight(o: &Operands) -> f32 {
    if !o.percents.is_empty() && o.has_of && !o.numbers.is_empty() {
        o.percents.len() as f32 * o.numbers.len() as f32
    } else {
        0.0
    }
}

fn single_weight(o: &Operands, n_values: usize) -> f32 {
    // A lone-value echo only applies with no relational cues. `%`, `of`, or
    // `average` all imply a relationship between operands, not a bare value.
    let bare_value = o.ops.is_empty()
        && o.quantities.is_empty()
        && o.percents.is_empty()
        && !o.has_of
        && !o.has_avg;
    if n_values == 1 && bare_value {
        1.0
    } else {
        0.0
    }
}

fn value_count(o: &Operands) -> usize {
    if !o.numbers.is_empty() {
        o.numbers.len()
    } else {
        o.prices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Currency;
    use crate::extract::extract;
    use crate::lexer::lex;

    fn ops(s: &str) -> Operands {
        extract(&lex(s))
    }

    #[test]
    fn spec_input_is_well_specified() {
        let o = ops("I buy persona5 3time each item 20$ in what is price total");
        assert_eq!(score(&o), 0.0);
        assert!(matches!(
            decision(&o, &UnderspecConfig::default()),
            QuestBenchDecision::PlanSkip
        ));
    }

    #[test]
    fn two_bare_numbers_are_underspecified() {
        let o = ops("20 30");
        assert_eq!(score(&o), 1.0);
        assert!(needs_clarification(&o, &UnderspecConfig::default()));
    }

    #[test]
    fn plain_single_value_is_well_specified() {
        let o = ops("20$");
        assert_eq!(score(&o), 0.0);
        let _ = Currency::Dollar;
    }

    #[test]
    fn average_input_routes_to_hot_tier() {
        let o = ops("average of 4 8 and 12");
        let cfg = UnderspecConfig::default();
        assert!(!needs_clarification(&o, &cfg));
        assert!(matches!(tier(&o, &cfg), MemoryTier::Hot));
    }

    #[test]
    fn percent_of_routes_to_hot_tier() {
        let o = ops("20% of 50");
        let cfg = UnderspecConfig::default();
        assert!(!needs_clarification(&o, &cfg));
        assert!(matches!(tier(&o, &cfg), MemoryTier::Hot));
    }
}
