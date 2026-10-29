//! Notebook 10's GRPO, the parts that are arithmetic on the host: what an episode's update reads,
//! its return and advantage, the minibatch's normalisation, and the error states a step leaves
//! for later steps to continue.
//!
//! - Return: the episode's err0 score minus `kl_coef` times its summed per-token KL to the
//!   reference (k1: `log π_old − log π_ref` over its policy tokens).
//! - Advantage: the return minus its group's mean (no division by the group's spread), on every
//!   actor token of the episode.
//! - Each minibatch's advantages are z-scored over its actor tokens (minus the mean, over the
//!   Bessel standard deviation plus 1e-8), as 08.5's `adv_norm minibatch`.
//! - Actor tokens: the policy's tokens less each turn's closing `<|im_end|>` and the answer's
//!   scaffolding (`freeze_turn_ends`, `freeze_answer_structure`), as 09.

use crate::membrane::batch::Rows;
use crate::membrane::episode::{Episode, PrefilledTurn, Start};
use crate::membrane::pyjson::Value;
use crate::membrane::tools::{self, CALL_BUDGET_ERROR, COMPARE_BUDGET_ERROR, CONVERT_BUDGET_ERROR};
use crate::tree::Reads;

/// What the update reads off one episode: every policy token (the KL and the old log-probs) and
/// which of them the actor's loss takes.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// Positions in the whole sequence (prompt and tokens) whose hidden state predicts each
    /// policy token.
    pub reads: Reads,
    pub actor: Vec<bool>,
}

pub fn prepare(ep: &Episode, rows: &Rows) -> Prepared {
    let p = ep.prompt_ids.len();
    let n = rows.mask.iter().filter(|&&m| m > 0.0).count();
    let mut reads = Reads::default();
    let mut actor = Vec::with_capacity(n);
    for k in 0..n {
        let i = rows.index[k];
        reads.at.push(p + i - 1);
        reads.targets.push(ep.ids[i]);
        let keep = (1.0 - rows.turn_end[k]) * (1.0 - rows.answer[k] + rows.content[k]);
        actor.push(keep > 0.0);
    }
    Prepared { reads, actor }
}

/// The episode's summed k1 KL over its policy tokens.
pub fn kl_sum(old: &[f32], reference: &[f32]) -> f64 {
    old.iter().zip(reference).map(|(a, b)| (*a as f64) - (*b as f64)).sum()
}

/// Each return minus its group's mean.
pub fn group_advantages(returns: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    let mut adv = vec![0.0; returns.len()];
    for g in groups {
        let mean = g.iter().map(|&e| returns[e]).sum::<f64>() / g.len() as f64;
        for &e in g {
            adv[e] = returns[e] - mean;
        }
    }
    adv
}

/// The minibatch's per-token advantages: each episode's advantage on its actor tokens, z-scored
/// over all of them (unchanged when fewer than two), in f32 as the loss takes them. 08.5
/// computes this in float32 over a padded batch; here the sums are f64 and only the result f32.
pub fn normalized(episode_adv: &[f64], actor_counts: &[usize]) -> Vec<f32> {
    let n: usize = actor_counts.iter().sum();
    if n < 2 {
        return episode_adv.iter().map(|&a| a as f32).collect();
    }
    let total: f64 = episode_adv
        .iter()
        .zip(actor_counts)
        .map(|(&a, &c)| a * c as f64)
        .sum();
    let mean = total / n as f64;
    let var = episode_adv
        .iter()
        .zip(actor_counts)
        .map(|(&a, &c)| (a - mean).powi(2) * c as f64)
        .sum::<f64>()
        / (n - 1) as f64;
    let std = var.sqrt();
    episode_adv
        .iter()
        .map(|&a| ((a - mean) / (std + 1e-8)) as f32)
        .collect()
}

fn budget_error(result: Option<&str>) -> bool {
    matches!(result, Some(r) if r == CALL_BUDGET_ERROR || r == CONVERT_BUDGET_ERROR || r == COMPARE_BUDGET_ERROR)
}

/// The episode's first error state: its turns up to and including the turn of its first call
/// that came back an error (a call over its budget does not count), with those turns' calls and
/// replies; the episode continues from the reply. The policy's tokens count against the new
/// episode's budgets as they did against this one's. None when no call errored, or when the
/// state leaves no token budget.
pub fn error_state(ep: &Episode, max_new_tokens: usize) -> Option<Start> {
    let first = ep
        .calls
        .iter()
        .find(|c| tools::result_is_error(c.result.as_deref()) && !budget_error(c.result.as_deref()))?;
    let last_turn = first.turn?;
    let mut turns = Vec::with_capacity(last_turn + 1);
    let mut spent = 0;
    for t in 0..=last_turn {
        let start = ep.turn_starts[t];
        let written: Vec<u32> = ep.ids[start..]
            .iter()
            .zip(&ep.policy[start..])
            .take_while(|(_, mine)| **mine)
            .map(|(&id, _)| id)
            .collect();
        spent += written.len();
        let calls = ep
            .calls
            .iter()
            .filter(|c| c.turn == Some(t))
            .map(|c| tools::Call { turn: None, ..c.clone() })
            .collect();
        turns.push(PrefilledTurn {
            text: ep.turns[t].clone(),
            written,
            calls,
        });
    }
    (spent < max_new_tokens).then_some(Start {
        turns,
        tokens_count: true,
    })
}

/// A state kept for later steps.
#[derive(Debug, Clone)]
pub struct ErrorState {
    pub case: Value,
    pub start: Start,
    pub step: usize,
    pub uses: usize,
    /// The id of the case and the episode's index within its step, to name it in the records.
    pub origin: String,
}

/// SplitMix64: the run's draws (questions, error states, minibatch order), seeded.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advantages_and_normalisation_follow_the_formulas() {
        let returns = [0.9, 0.1, 0.5, 0.5, 1.0, 0.0, 0.25, 0.75];
        let groups = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let adv = group_advantages(&returns, &groups);
        assert!((adv[0] - 0.4).abs() < 1e-12 && (adv[1] + 0.4).abs() < 1e-12);
        assert!((adv[2]).abs() < 1e-12 && (adv[4] - 0.5).abs() < 1e-12);
        // z-score over tokens: episode e repeated counts[e] times.
        let counts = [3usize, 1, 0, 2, 5, 4, 1, 1];
        let z = normalized(&adv, &counts);
        let tokens: Vec<f64> = adv
            .iter()
            .zip(&counts)
            .flat_map(|(&a, &c)| std::iter::repeat_n(a, c))
            .collect();
        let n = tokens.len() as f64;
        let mean = tokens.iter().sum::<f64>() / n;
        let std = (tokens.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        for (e, &a) in adv.iter().enumerate() {
            let want = (a - mean) / (std + 1e-8);
            assert!(((z[e] as f64) - want).abs() < 1e-6, "episode {e}");
        }
    }

    #[test]
    fn rng_is_deterministic_and_shuffles_all() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut v: Vec<usize> = (0..8).collect();
        a.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }
}
