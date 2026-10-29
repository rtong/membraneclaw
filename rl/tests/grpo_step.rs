//! One GRPO step's arithmetic against the formulas written out independently in f64: group
//! advantages, the minibatch z-score, PPO's clipped loss and its gradient, gradient clipping and
//! two AdamW steps (eps 1e-5, bias-corrected).

use burn::module::{Module, Param};
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Tensor, TensorData};
use dash_train::clip::{ClipGradNormOptions, clip_grad_norm};
use dash_train::optim::{AdamW, Beta1};
use membrane_rl::grpo;
use membrane_rl::tree::ppo_loss;

#[derive(Module, Debug)]
struct Toy {
    lp: Param<Tensor<1>>,
}

fn host(t: Tensor<1>) -> Vec<f64> {
    t.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

#[test]
fn one_step_follows_the_formulas() {
    let device = Device::flex().autodiff();
    let clip_eps = 0.2f64;
    // max_norm below the gradient's norm, so that clipping acts.
    let (lr, eps, b1, b2, max_norm) = (2e-6f64, 1e-5f64, 0.9f64, 0.999f64, 0.1f64);

    // Two groups of two episodes; returns already less their KL.
    let returns = [0.8, 0.2, 0.5, 0.9];
    let groups = vec![vec![0, 1], vec![2, 3]];
    let counts = [3usize, 2, 4, 1];
    let adv = grpo::group_advantages(&returns, &groups);
    let z = grpo::normalized(&adv, &counts);

    // f64: the group means, then the z-score over the 10 actor tokens.
    let means = [0.5, 0.5, 0.7, 0.7];
    let a64: Vec<f64> = returns.iter().zip(means).map(|(r, m)| r - m).collect();
    let tokens: Vec<f64> = a64
        .iter()
        .zip(counts)
        .flat_map(|(&a, c)| std::iter::repeat_n(a, c))
        .collect();
    let n = tokens.len() as f64;
    let mu = tokens.iter().sum::<f64>() / n;
    let sd = (tokens.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let z64: Vec<f64> = a64.iter().map(|a| (a - mu) / (sd + 1e-8)).collect();
    for e in 0..4 {
        assert!((z[e] as f64 - z64[e]).abs() <= 1e-6, "z-score of episode {e}");
    }

    // Per token: old log-probs, new ones (some outside the clip range), the advantage.
    let old: Vec<f32> = vec![-1.2, -0.3, -2.0, -0.7, -0.05, -1.5, -0.9, -3.0, -0.2, -0.6];
    let new: Vec<f32> = vec![-1.0, -0.5, -1.6, -0.7, -0.01, -1.9, -0.85, -2.5, -0.4, -0.61];
    let owner: Vec<usize> = (0..4).flat_map(|e| std::iter::repeat_n(e, counts[e])).collect();
    let mut toy = Toy {
        lp: Param::from_tensor(Tensor::from_data(TensorData::new(new.clone(), [10]), &device)),
    };
    let mut optim = ModuleOptimizer::from(AdamW {
        beta1: Beta1::new(b1),
        beta2: b2,
        eps,
        weight_decay: 0.0,
    });
    let mut p64: Vec<f64> = new.iter().map(|&x| x as f64).collect();
    let (mut m64, mut v64) = (vec![0.0f64; 10], vec![0.0f64; 10]);
    for t in 1..=2 {
        // The minibatch's loss: the sum over its episodes of each one's token-mean surrogate,
        // over the number of episodes.
        let lp = toy.lp.val();
        let mut loss: Option<Tensor<1>> = None;
        let mut start = 0;
        for e in 0..4 {
            let c = counts[e];
            let (l, _) = ppo_loss(
                lp.clone().narrow(0, start, c),
                &old[start..start + c],
                &vec![z[e]; c],
                1.0 / (c as f32 * 4.0),
                clip_eps as f32,
            );
            loss = Some(match loss {
                Some(x) => x + l,
                None => l,
            });
            start += c;
        }
        let loss = loss.unwrap();
        let mut grads = GradientsParams::from_grads(loss.backward(), &toy);
        let norm = host(clip_grad_norm(
            &toy,
            &mut grads,
            &ClipGradNormOptions {
                max_norm,
                ..ClipGradNormOptions::default()
            },
        ))[0];
        toy = optim.step(lr, toy, grads);

        // f64: the gradient of -min(r A, clip(r) A) / (c * 4) with respect to the log-prob.
        let g64: Vec<f64> = (0..10)
            .map(|k| {
                let e = owner[k];
                let r = (p64[k] - old[k] as f64).exp();
                let a = z64[e];
                let clipped = r.clamp(1.0 - clip_eps, 1.0 + clip_eps) * a;
                let active = r * a <= clipped || (r >= 1.0 - clip_eps && r <= 1.0 + clip_eps);
                if active { -a * r / (counts[e] as f64 * 4.0) } else { 0.0 }
            })
            .collect();
        let total = g64.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!((norm - total).abs() <= 1e-6 * total.max(1.0), "step {t}: norm {norm} against {total}");
        let coef = (max_norm / (total + 1e-6)).min(1.0);
        for k in 0..10 {
            let g = g64[k] * coef;
            m64[k] = b1 * m64[k] + (1.0 - b1) * g;
            v64[k] = b2 * v64[k] + (1.0 - b2) * g * g;
            let mh = m64[k] / (1.0 - b1.powi(t));
            let vh = v64[k] / (1.0 - b2.powi(t));
            p64[k] -= lr * mh / (vh.sqrt() + eps);
        }
        let got = host(toy.lp.val().inner());
        for k in 0..10 {
            let moved64 = p64[k] - new[k] as f64;
            let moved = got[k] - new[k] as f64;
            // f32 parameters: compare the step taken, to the float's resolution at the value.
            let ulp = (new[k].abs() as f64) * f32::EPSILON as f64;
            assert!(
                (moved - moved64).abs() <= 1e-6 * moved64.abs() + ulp,
                "step {t}, token {k}: moved {moved:e} against {moved64:e}"
            );
        }
        println!("step {t}: grad norm {norm:.6e} (f64 {total:.6e}); parameters within f32 resolution");
    }
}
