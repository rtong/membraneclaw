//! The rollout policy: the model writing the next turn of a batch of episodes, one token per
//! step for every unfinished row, sampled on the device.
//!
//! No prefix runs twice. The generator keeps the state after every token sequence it has run:
//! a call's input continues the longest kept sequence it begins with, and only the rest is
//! prefilled. Within a call, rows with the same input are prefilled once and forked; when
//! several inputs share a long common prefix (the system turn and the tool declarations, at a
//! step's first turn), it is prefilled once first. Between calls each row keeps its own state,
//! so a turn after a tool reply prefills only the reply. The kept states belong to the weights
//! that made them: `set_model` drops them.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::time::Instant;

use burn::tensor::{Int, Tensor, TensorData};
use dash_ops::sample::{SampleOptions, sample};

use crate::membrane::episode::{Episode, Harness, Policy, Running, Start};
use crate::membrane::pyjson::Value;
use crate::membrane::tools::{Call, Tools};
use crate::qwen3_5::{Batch, Decoder, Qwen3_5ForCausalLM, RotaryTable, State};

struct Entry {
    tokens: Vec<u32>,
    state: State,
    /// The hidden state after the last token, `[1, hidden]`, when a call may start from it.
    last: Option<Tensor<2>>,
    /// Kept across calls (a common prefix); a row's entry lasts until the next call.
    shared: bool,
}

/// What a generator did, summed over its calls.
#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub calls: usize,
    pub prefill_tokens: usize,
    pub prefills: usize,
    pub decode_steps: usize,
    pub decoded_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

pub struct Generator<'h> {
    model: Qwen3_5ForCausalLM,
    decoder: Decoder,
    table: RotaryTable,
    harness: &'h Harness,
    options: SampleOptions,
    entries: Vec<Entry>,
    /// A common prefix this much longer than what the inputs already continue is prefilled once.
    pub min_shared: usize,
    /// The key/value positions a continuous rollout's batch is given from the start.
    pub capacity: usize,
    pub stats: GenStats,
    /// The last call's rows: each written token's log-prob under the sampling distribution.
    pub logprobs: Vec<Vec<f32>>,
}

impl<'h> Generator<'h> {
    /// `model` runs without autodiff; `options` sets the temperature and the seed (the offset
    /// advances one per decode step, so every draw of a run is its own).
    pub fn new(
        model: Qwen3_5ForCausalLM,
        table: RotaryTable,
        harness: &'h Harness,
        mut options: SampleOptions,
    ) -> Self {
        if !options.masked.contains(&harness.suppress) {
            options.masked.push(harness.suppress);
        }
        Self {
            decoder: Decoder::new(&model),
            model,
            table,
            harness,
            options,
            entries: Vec::new(),
            min_shared: 64,
            capacity: 0,
            stats: GenStats::default(),
            logprobs: Vec::new(),
        }
    }

    /// New weights: every kept state was made by the old ones.
    pub fn set_model(&mut self, model: Qwen3_5ForCausalLM) {
        self.decoder.set_lora(&model);
        self.model = model;
        self.entries.clear();
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        self.options.temperature = temperature;
    }

    /// The sampler's temperature, seed and offset.
    pub fn sampling(&self) -> (f32, u64, u64) {
        (self.options.temperature, self.options.seed, self.options.offset)
    }

    pub fn set_sampling(&mut self, temperature: f32, seed: u64, offset: u64) {
        self.options.temperature = temperature;
        self.options.seed = seed;
        self.options.offset = offset;
    }

    pub fn restore_sampling(&mut self, (temperature, seed, offset): (f32, u64, u64)) {
        self.set_sampling(temperature, seed, offset);
    }

    pub fn model(&self) -> &Qwen3_5ForCausalLM {
        &self.model
    }

    /// Forgets every kept state.
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// The kept entry `input` continues furthest: its tokens a prefix of `input`, and shorter
    /// unless the entry holds the hidden state that predicts the next token.
    fn best(&self, input: &[u32]) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.tokens.len();
            let fits = n <= input.len()
                && (n < input.len() || e.last.is_some())
                && input[..n] == e.tokens[..];
            if fits && best.is_none_or(|(_, m)| n > m) {
                best = Some((i, n));
            }
        }
        best.map(|(i, _)| i)
    }

    /// The state after `tokens` and the hidden state after its last token, prefilling what the
    /// entry `from` (if any) has not run.
    fn prefill(&mut self, from: Option<usize>, tokens: &[u32]) -> (State, Tensor<2>) {
        let (mut state, done) = match from {
            Some(i) => (self.entries[i].state.clone(), self.entries[i].tokens.len()),
            None => (self.model.empty_state(), 0),
        };
        if done == tokens.len() {
            let last = self.entries[from.unwrap()].last.clone().expect("best() checks");
            return (state, last);
        }
        let t0 = Instant::now();
        let rest: Vec<i64> = tokens[done..].iter().map(|&t| t as i64).collect();
        let n = rest.len();
        let device = self.model.model.norm.weight.device();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(rest, [1, n]), &device);
        let hidden = self.model.forward(ids, &self.table, &mut state);
        let d = hidden.dims()[2];
        let last = hidden.narrow(1, n - 1, 1).reshape([1, d]);
        self.stats.prefill_tokens += n;
        self.stats.prefills += 1;
        self.stats.prefill_seconds += t0.elapsed().as_secs_f64();
        (state, last)
    }

    fn row_copy(&self, batch: &Batch, b: usize) -> State {
        // A copy of the row, so that the batch's buffers are not kept alive by it.
        self.model.row_copy(batch, b)
    }
}

/// The length of the longest common prefix of `seqs`.
fn common_prefix(seqs: &[&Vec<u32>]) -> usize {
    let Some(first) = seqs.first() else { return 0 };
    let mut n = first.len();
    for s in &seqs[1..] {
        n = n.min(s.iter().zip(first.iter()).take_while(|(a, b)| a == b).count());
    }
    n
}

impl Policy for Generator<'_> {
    fn generate(&mut self, inputs: &[Vec<u32>], budgets: &[usize]) -> Vec<Vec<u32>> {
        self.stats.calls += 1;
        let rows = inputs.len();
        if rows == 0 {
            return Vec::new();
        }
        // Rows with the same input share its prefill.
        let mut uniques: Vec<&Vec<u32>> = Vec::new();
        let mut which: HashMap<&Vec<u32>, usize> = HashMap::new();
        let unique_of: Vec<usize> = inputs
            .iter()
            .map(|x| {
                *which.entry(x).or_insert_with(|| {
                    uniques.push(x);
                    uniques.len() - 1
                })
            })
            .collect();
        // A long prefix common to the inputs runs once.
        if uniques.len() >= 2 {
            let lcp = common_prefix(&uniques);
            let gain = uniques
                .iter()
                .filter(|u| {
                    let have = self.best(u).map_or(0, |i| self.entries[i].tokens.len());
                    have + self.min_shared <= lcp
                })
                .count();
            if gain >= 2 {
                let prefix = uniques[0][..lcp].to_vec();
                let from = self.best(&prefix);
                let (state, last) = self.prefill(from, &prefix);
                self.entries.push(Entry {
                    tokens: prefix,
                    state,
                    last: Some(last),
                    shared: true,
                });
            }
        }
        let mut starts: Vec<(State, Tensor<2>)> = Vec::with_capacity(uniques.len());
        for u in &uniques {
            let from = self.best(u);
            starts.push(self.prefill(from, u));
        }
        // The rows' states from the last call are continued now; only the common prefixes stay.
        self.entries.retain(|e| e.shared);

        let t0 = Instant::now();
        let h = self.harness;
        let device = self.model.model.norm.weight.device();
        let states: Vec<&State> = unique_of.iter().map(|&u| &starts[u].0).collect();
        let mut batch = self.model.batch(&states);
        let mut hidden = Tensor::cat(
            unique_of.iter().map(|&u| starts[u].1.clone()).collect(),
            0,
        );
        let uniques_len = starts.len();
        drop(starts);
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); rows];
        let mut lps: Vec<Vec<f32>> = vec![Vec::new(); rows];
        let mut text: Vec<String> = vec![String::new(); rows];
        let mut finals: Vec<Option<State>> = (0..rows).map(|_| None).collect();
        loop {
            let logits = self.model.logits(hidden);
            let drawn = sample(logits, &self.options);
            self.options.offset += 1;
            self.stats.decode_steps += 1;
            let tokens: Vec<i64> = drawn
                .tokens
                .into_data()
                .convert::<i64>()
                .try_into_vec::<i64>()
                .expect("token ids");
            let logprob: Vec<f32> = drawn
                .sampling_logprob
                .into_data()
                .convert::<f32>()
                .try_into_vec::<f32>()
                .expect("log-probs");
            let mut next: Vec<i64> = vec![h.im_end as i64; rows];
            let mut live_now = 0;
            for p in 0..rows {
                if !batch.active[p] {
                    continue;
                }
                let r = p;
                let t = tokens[p] as u32;
                out[r].push(t);
                lps[r].push(logprob[p]);
                self.stats.decoded_tokens += 1;
                let mut stop = t == h.im_end || out[r].len() >= budgets[r];
                if !stop && h.stop_at_json {
                    let piece = h.piece(t);
                    text[r].push_str(&piece);
                    stop = piece.contains('}') && h.answer_written(&text[r]);
                }
                if stop {
                    // The row's state covers its input and every token but the last.
                    finals[r] = Some(self.row_copy(&batch, p));
                    batch.active[p] = false;
                } else {
                    next[p] = tokens[p];
                    live_now += 1;
                }
            }
            if live_now == 0 {
                break;
            }
            let tokens = Tensor::<1, Int>::from_data(TensorData::new(next, [rows]), &device);
            hidden = self.decoder.decode(&self.model, tokens, &self.table, &mut batch);
        }
        self.stats.decode_seconds += t0.elapsed().as_secs_f64();
        if std::env::var_os("GEN_VERBOSE").is_some() {
            eprintln!(
                "    generate: {rows} rows ({} inputs), {:.1}s decoding, longest turn {}",
                uniques_len,
                t0.elapsed().as_secs_f64(),
                out.iter().map(Vec::len).max().unwrap_or(0)
            );
        }
        // Kept for the next call: the common prefixes, and each row's state.
        for (r, state) in finals.into_iter().enumerate() {
            let state = state.expect("every row stops");
            let mut tokens = inputs[r].clone();
            tokens.extend_from_slice(&out[r][..out[r].len() - 1]);
            debug_assert_eq!(tokens.len(), state.seen);
            self.entries.push(Entry {
                tokens,
                state,
                last: None,
                shared: false,
            });
        }
        self.logprobs = lps;
        out
    }
}

/// A batch row's episode while it writes a turn.
struct Writing {
    episode: usize,
    out: Vec<u32>,
    logprobs: Vec<f32>,
    text: String,
    budget: usize,
}

impl Generator<'_> {
    /// The episodes of `cases`, each from its `start`, run continuously: `slots` rows decode
    /// together; a row whose turn ends runs its calls on a thread while the others decode, and
    /// writes its next turn as soon as they answer; a finished episode's row takes the next
    /// waiting episode. Each episode's turns, calls and budgets are `rollout_from`'s; only the
    /// order episodes advance in differs. Returns the episodes, and each one's written tokens'
    /// log-probs under the sampling distribution.
    pub fn rollout(
        &mut self,
        cases: &[Value],
        starts: &[Start],
        tools: &mut (dyn Tools + Send),
        slots: usize,
    ) -> Result<(Vec<Episode>, Vec<Vec<f32>>), String> {
        let h = self.harness;
        let n = cases.len();
        let mut runs: Vec<Running> = cases
            .iter()
            .zip(starts)
            .map(|(c, s)| h.begin(c, s))
            .collect::<Result<_, _>>()?;
        let mut lps: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut queue: VecDeque<usize> = (0..n).collect();
        let device = self.model.model.norm.weight.device();
        let t_all = Instant::now();

        // The first rows: their inputs prefilled with the prefix cache (common prefixes once,
        // identical inputs once).
        let mut first: Vec<(usize, usize)> = Vec::new(); // (episode, budget)
        while first.len() < slots {
            let Some(e) = queue.pop_front() else { break };
            let b = runs[e].budget(h);
            if b > 0 {
                first.push((e, b));
            }
        }
        if first.is_empty() {
            return Ok((runs.into_iter().map(|r| r.ep).collect(), lps));
        }
        let inputs: Vec<Vec<u32>> = first.iter().map(|&(e, _)| runs[e].input()).collect();
        let mut uniques: Vec<&Vec<u32>> = Vec::new();
        let mut which: HashMap<&Vec<u32>, usize> = HashMap::new();
        let unique_of: Vec<usize> = inputs
            .iter()
            .map(|x| {
                *which.entry(x).or_insert_with(|| {
                    uniques.push(x);
                    uniques.len() - 1
                })
            })
            .collect();
        // Every input of the call, those waiting included, shares the system turn: run it once.
        let all_inputs: Vec<Vec<u32>> = (0..n).map(|e| runs[e].input()).collect();
        let all_refs: Vec<&Vec<u32>> = all_inputs.iter().collect();
        let lcp = common_prefix(&all_refs);
        if n >= 2 && lcp >= self.min_shared {
            let have = self.best(&all_inputs[0][..lcp]).map_or(0, |i| self.entries[i].tokens.len());
            if have < lcp {
                let prefix = all_inputs[0][..lcp].to_vec();
                let from = self.best(&prefix);
                let (state, last) = self.prefill(from, &prefix);
                self.entries.push(Entry {
                    tokens: prefix,
                    state,
                    last: Some(last),
                    shared: true,
                });
            }
        }
        drop(all_refs);
        let mut starts_u: Vec<(State, Tensor<2>)> = Vec::with_capacity(uniques.len());
        for u in &uniques {
            let from = self.best(u);
            starts_u.push(self.prefill(from, u));
        }
        let states: Vec<&State> = unique_of.iter().map(|&u| &starts_u[u].0).collect();
        let mut batch = self.model.batch_sized(&states, self.capacity);
        let mut hidden = Tensor::cat(
            unique_of.iter().map(|&u| starts_u[u].1.clone()).collect(),
            0,
        );
        drop(states);
        drop(starts_u);
        let rows = first.len();
        let d = hidden.dims()[1];
        let mut rowing: Vec<Option<Writing>> = first
            .iter()
            .map(|&(e, budget)| {
                Some(Writing {
                    episode: e,
                    out: Vec::new(),
                    logprobs: Vec::new(),
                    text: String::new(),
                    budget,
                })
            })
            .collect();

        let (job_tx, job_rx) = mpsc::channel::<(usize, Vec<(String, Value)>)>();
        let (res_tx, res_rx) = mpsc::channel::<(usize, Vec<String>)>();
        let result: Result<(), String> = std::thread::scope(|s| {
            s.spawn(move || {
                for (e, jobs) in job_rx {
                    let answers = tools.call_many(&jobs);
                    if res_tx.send((e, answers)).is_err() {
                        break;
                    }
                }
            });
            // Episodes whose calls are out: their row, the calls, which of them the tools run.
            let mut waiting: HashMap<usize, (usize, Vec<Call>, Vec<usize>)> = HashMap::new();
            // Episodes ready to write again (their row), and rows free for a waiting episode.
            let mut ready: Vec<(usize, usize)> = Vec::new();
            let mut free: Vec<usize> = Vec::new();
            let mut timing = [0.0f64; 4];
            loop {
                // 1. Tool replies in (waiting for one when no row is decoding).
                let decoding = batch.active.iter().any(|&a| a);
                if !decoding && ready.is_empty() && free.is_empty() {
                    if waiting.is_empty() {
                        break;
                    }
                    let (e, answers) = res_rx.recv().map_err(|_| "the tool thread died".to_string())?;
                    take_answers(h, &mut runs, &mut waiting, &mut ready, e, answers);
                }
                while let Ok((e, answers)) = res_rx.try_recv() {
                    take_answers(h, &mut runs, &mut waiting, &mut ready, e, answers);
                }
                // 2. Episodes continue: their new tokens prefilled from their row's state.
                let ts = Instant::now();
                for (e, r) in std::mem::take(&mut ready) {
                    let budget = runs[e].budget(h);
                    if budget == 0 {
                        free.push(r);
                        continue;
                    }
                    let input = runs[e].input();
                    let fed = batch.lengths[r];
                    let mut state = self.model.row(&batch, r);
                    let last = self.prefill_from(&mut state, &input[fed..], &device);
                    self.model.write_row(&mut batch, r, &state, fed);
                    drop(state);
                    hidden = hidden.slice_assign([r..r + 1, 0..d], last);
                    rowing[r] = Some(Writing {
                        episode: e,
                        out: Vec::new(),
                        logprobs: Vec::new(),
                        text: String::new(),
                        budget,
                    });
                }
                // 3. Free rows take waiting episodes.
                for r in std::mem::take(&mut free) {
                    let mut started = false;
                    while let Some(e) = queue.pop_front() {
                        let budget = runs[e].budget(h);
                        if budget == 0 {
                            continue;
                        }
                        let input = runs[e].input();
                        let from = self.best(&input);
                        let (state, last) = self.prefill(from, &input);
                        self.model.write_row(&mut batch, r, &state, 0);
                        hidden = hidden.slice_assign([r..r + 1, 0..d], last);
                        rowing[r] = Some(Writing {
                            episode: e,
                            out: Vec::new(),
                            logprobs: Vec::new(),
                            text: String::new(),
                            budget,
                        });
                        started = true;
                        break;
                    }
                    if !started {
                        rowing[r] = None;
                        batch.active[r] = false;
                    }
                }
                timing[3] += ts.elapsed().as_secs_f64();
                if !batch.active.iter().any(|&a| a) {
                    continue;
                }
                // 4. One token for every decoding row.
                let ts = Instant::now();
                let logits = self.model.logits(hidden.clone());
                let drawn = sample(logits, &self.options);
                self.options.offset += 1;
                self.stats.decode_steps += 1;
                let tokens: Vec<i64> = drawn
                    .tokens
                    .into_data()
                    .convert::<i64>()
                    .try_into_vec::<i64>()
                    .expect("token ids");
                let logprob: Vec<f32> = drawn
                    .sampling_logprob
                    .into_data()
                    .convert::<f32>()
                    .try_into_vec::<f32>()
                    .expect("log-probs");
                timing[0] += ts.elapsed().as_secs_f64();
                let ts = Instant::now();
                let mut next: Vec<i64> = vec![h.im_end as i64; rows];
                for r in 0..rows {
                    if !batch.active[r] {
                        continue;
                    }
                    let w = rowing[r].as_mut().expect("a decoding row writes");
                    let t = tokens[r] as u32;
                    w.out.push(t);
                    w.logprobs.push(logprob[r]);
                    self.stats.decoded_tokens += 1;
                    let mut stop = t == h.im_end || w.out.len() >= w.budget;
                    if !stop && h.stop_at_json {
                        let piece = h.piece(t);
                        w.text.push_str(&piece);
                        stop = piece.contains('}') && h.answer_written(&w.text);
                    }
                    if !stop {
                        next[r] = tokens[r];
                        continue;
                    }
                    batch.active[r] = false;
                    let w = rowing[r].take().unwrap();
                    let e = w.episode;
                    lps[e].extend(&w.logprobs);
                    match runs[e].take_turn(h, &w.out, w.budget)? {
                        None => free.push(r),
                        Some((calls, ks)) if ks.is_empty() => {
                            runs[e].answer(h, calls);
                            ready.push((e, r));
                        }
                        Some((calls, ks)) => {
                            let jobs = ks
                                .iter()
                                .map(|&k| (calls[k].name.clone().unwrap(), calls[k].arguments.clone().unwrap()))
                                .collect();
                            job_tx.send((e, jobs)).map_err(|_| "the tool thread died".to_string())?;
                            waiting.insert(e, (r, calls, ks));
                        }
                    }
                }
                timing[1] += ts.elapsed().as_secs_f64();
                if batch.active.iter().any(|&a| a) {
                    let ts = Instant::now();
                    let tokens = Tensor::<1, Int>::from_data(TensorData::new(next, [rows]), &device);
                    hidden = self.decoder.decode(&self.model, tokens, &self.table, &mut batch);
                    timing[2] += ts.elapsed().as_secs_f64();
                }
            }
            drop(job_tx);
            if std::env::var_os("GEN_VERBOSE").is_some() {
                eprintln!(
                    "    rollout: sample+read {:.1}s, rows {:.1}s, decode dispatch {:.1}s, insertions {:.1}s",
                    timing[0], timing[1], timing[2], timing[3]
                );
            }
            Ok(())
        });
        result?;
        self.stats.calls += 1;
        self.stats.decode_seconds += t_all.elapsed().as_secs_f64();
        self.entries.retain(|e| e.shared);
        Ok((runs.into_iter().map(|r| r.ep).collect(), lps))
    }

    /// Runs `tokens` continuing `state`; the hidden state after the last, `[1, hidden]`.
    fn prefill_from(&mut self, state: &mut State, tokens: &[u32], device: &burn::tensor::Device) -> Tensor<2> {
        let t0 = Instant::now();
        let n = tokens.len();
        let v: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(v, [1, n]), device);
        let hidden = self.model.forward(ids, &self.table, state);
        let d = hidden.dims()[2];
        self.stats.prefill_tokens += n;
        self.stats.prefills += 1;
        self.stats.prefill_seconds += t0.elapsed().as_secs_f64();
        hidden.narrow(1, n - 1, 1).reshape([1, d])
    }
}

/// A reply from the tools: the episode's calls answered, and the episode ready to write on.
fn take_answers(
    h: &Harness,
    runs: &mut [Running],
    waiting: &mut HashMap<usize, (usize, Vec<Call>, Vec<usize>)>,
    ready: &mut Vec<(usize, usize)>,
    e: usize,
    answers: Vec<String>,
) {
    let (r, mut calls, ks) = waiting.remove(&e).expect("an answer for a waiting episode");
    for (k, a) in ks.into_iter().zip(answers) {
        calls[k].result = Some(a);
    }
    runs[e].answer(h, calls);
    ready.push((e, r));
}
