# CELAR: Cross-Embodiment Latent Action Reinforcement Learning with Executable Robot-Specific Decoders

## Paper Outline

**Status:** Detailed outline, not a submission-ready manuscript  
**Evidence basis:** `attm5` CELAR pilot results, C0 MuJoCo data audit, and planned multi-embodiment validation  
**Date:** 2026-11-09

---

## 1. Working Titles

### Preferred title

**CELAR: Cross-Embodiment Latent Action Reinforcement Learning with Executable Robot-Specific Decoders**

### Alternative titles

1. **Learning a Shared Action Semantics for Heterogeneous Robot Arms**
2. **Executable Cross-Embodiment Latent Actions for Reinforcement Learning**
3. **Do Shared Latent Actions Improve Robot RL Sample Efficiency?**
4. **From Paired Demonstrations to Cross-Embodiment Action Transfer**

The preferred title emphasizes both parts of the method: a shared latent action space and robot-specific executable decoders. Avoid titles that imply zero-shot deployment to unseen robots or real-world validation, because those claims are not yet established.

---

## 2. One-Sentence Thesis

**When the same manipulation task is observed across mechanically different robot arms, a compact shared latent action space learned from paired action chunks can preserve semantic correspondence across embodiments while robot-specific decoders maintain executable control, improving online reinforcement-learning efficiency over policies that learn native joint actions independently.**

---

## 3. Abstract Skeleton

### Motivation

Robot manipulation datasets and policies are fragmented by embodiment. Panda, UR5e, Sawyer, and other arms use different joint counts, action limits, kinematics, and control conventions, making direct policy or replay transfer difficult. Existing shared policies often condition a single policy on robot identity, but this does not guarantee that semantically similar actions occupy nearby latent positions or that decoded actions remain physically executable.

### Problem

Can a shared latent action representation:

1. align semantically similar action chunks across robots,
2. decode into safe native actions for each robot,
3. preserve single-robot control quality, and
4. improve online RL sample efficiency and few-shot adaptation?

### Method

Introduce **CELAR**, a cross-embodiment latent action learner. CELAR encodes observation, language/object context, native action chunks, and morphology descriptors into an 8-dimensional latent action. Each embodiment has an independent decoder that maps the shared latent action into its own joint or end-effector command space. Training combines:

- native action reconstruction,
- paired cross-embodiment alignment,
- cross-decoder cycle consistency,
- task-progress consequence consistency,
- temporal smoothness,
- morphology conditioning,
- deterministic safety projection.

### Evaluation

Evaluate on paired trajectories for Panda, UR5e, and Sawyer across pick, place, open-drawer, and push tasks. Compare native per-robot PPO, shared native actions, end-effector delta plus IK, action VAEs/VQ-VAEs, shared latent decoders, and CELAR ablations under matched data, online-step, and parameter budgets.

### Current preliminary evidence

In a controlled cross-embodiment proxy:

- Similar-action retrieval R@1 reaches 0.911 for CELAR versus 0.550 for a shared autoencoder.
- An equal-budget PPO pilot reaches 71.5% success for CELAR versus 37.5% for native-action PPO.
- A fixed body/task audit shows a +29.4 percentage-point mean improvement, with 83.3% of seed-body-task comparisons positive.
- A held-out Sawyer decoder reduces 5-shot semantic action error by 26% relative to training an action autoencoder from scratch.
- A MuJoCo C0 study produces 1,200 paired trajectories with zero split leakage, zero post-safety violations, 0.66-0.83 px RGB-D/EEP alignment, and 0.260 GB batch-8 peak GPU memory.

### Claim boundary

These are pilot results. The manuscript must state that the current online RL numbers come from a controlled proxy and require replication with vendor-grade MuJoCo/Robosuite assets, four tasks, three seeds, and external benchmarks before they can support a broad robotics claim.

---

## 4. Keywords

Cross-embodiment learning; latent action representation; reinforcement learning; behavior cloning; robot manipulation; action-chunk decoding; RGB-D; sample efficiency; few-shot adaptation; robot safety.

---

## 5. Introduction

### 5.1 Opening problem

Robot manipulation research increasingly spans multiple arm families, yet most policies remain embodiment-specific. A policy trained for a 7-DoF arm cannot be directly applied to a 6-DoF arm, and raw action replay is invalid because the dimensions, joint ordering, limits, and dynamics differ.

### 5.2 Why shared observations alone are insufficient

Shared visual or language encoders can represent the scene across robots, but the control interface remains heterogeneous. A scene encoder does not tell us:

- whether two different joint trajectories implement the same grasp,
- whether a shared action code is decodable by each robot,
- whether two robots produce the same task-level consequence from a latent action,
- whether the representation survives online policy optimization.

### 5.3 Gap

Prior work studies cross-embodiment policies, action abstraction, imitation, and latent action models, but rarely closes the loop across all of the following:

1. explicit paired action semantics,
2. robot-specific executable decoding,
3. safety constraints,
4. online RL efficiency,
5. new-embodiment adaptation.

### 5.4 Proposed insight

Separate what should be shared from what should remain embodiment-specific:

- shared: compact action semantics,
- specific: robot decoder and safety projection,
- optionally shared: high-level policy conditioned on morphology.

### 5.5 Contributions

State contributions conditionally, with the final wording depending on completed experiments:

1. **Representation contribution:** a paired cross-embodiment action objective that aligns action chunks by task phase while preserving robot-specific reconstruction.
2. **Executable-interface contribution:** robot-specific decoders plus deterministic safety projection that maintain action feasibility across different joint counts and limits.
3. **RL contribution:** evidence that a shared latent action interface can improve same-budget online policy learning relative to native-action PPO.
4. **Adaptation contribution:** a decoder-only adaptation protocol for an unseen embodiment using 5-20 demonstrations.
5. **Evaluation contribution:** a paired multi-embodiment data and evaluation protocol with explicit leakage, failure, perturbation, latency, and safety audits.

Do not list sample efficiency or cross-embodiment transfer as an established contribution until the MuJoCo/Robosuite replication is complete.

---

## 6. Related Work

### 6.1 Robot learning across embodiments

Cover:

- embodiment-conditioned policies,
- cross-embodiment datasets and demonstrations,
- morphology-aware networks,
- shared observation spaces with robot-specific heads,
- action normalization and end-effector interfaces.

Position CELAR as focusing on the *action interface*, not only the observation encoder.

### 6.2 Latent action models

Cover:

- action autoencoders and VAEs,
- VQ-VAE action tokenization,
- behavior cloning with compact action bottlenecks,
- latent action pretraining,
- action chunking and temporal abstraction.

Main distinction: CELAR aligns paired actions from mechanically different robots and retains a separately executable decoder per robot.

### 6.3 Cross-embodiment imitation and transfer

Discuss:

- observation-space alignment,
- domain adaptation,
- retargeting or inverse kinematics,
- shared-policy with embodiment IDs,
- adapter-based transfer.

Clearly separate CELAR from retargeting: CELAR does not require geometric retargeting at inference, although retargeting or IK can be a baseline.

### 6.4 Reinforcement learning with structured action spaces

Discuss:

- PPO in native joint space,
- action normalization,
- end-effector delta control,
- skill or option spaces,
- action priors from demonstrations,
- decoder-conditioned policies.

### 6.5 Manipulation benchmarks and data

Cover:

- MuJoCo/Robosuite,
- LIBERO/LIBERO-Plus,
- RoboTwin,
- CALVIN,
- RLBench/SimplerEnv if used later,
- paired and unpaired robot data construction.

---

## 7. Problem Formulation

### 7.1 Embodiments

Let \(k \in \{1,\ldots,K\}\) index embodiments. Each embodiment has:

- native action dimension \(A_k\),
- joint or end-effector control convention,
- joint limits \(l_k, u_k\),
- velocity and gripper constraints,
- morphology descriptor \(e_k\),
- decoder \(D_k\).

Use \(K=3\) in the main study:

- Panda: 7-DoF,
- UR5e: 6-DoF,
- Sawyer: 7-DoF.

### 7.2 Paired trajectories

For task \(T_j\), scene \(S_m\), and pair \(p\), collect trajectories:

\[
\tau_{k,p} = \{(I_t,D_t,q_{t,k},a_{t,k},g_{t,k},x_{t,k},s_{t,k})\}_{t=1}^{L}
\]

where \(I_t\) and \(D_t\) are RGB-D observations, \(q_{t,k}\) is proprioception, \(a_{t,k}\) is the native action, \(g_{t,k}\) is the gripper action, \(x_{t,k}\) is the EEF pose, and \(s_{t,k}\) is task progress/success.

All robots in a pair share:

- task ID,
- scene ID,
- initial object and goal state,
- instruction,
- task-phase labels.

Robots retain their own action dimensions and dynamics.

### 7.3 Action chunks

Define an action chunk of length \(H_a\):

\[
a_{t:t+H_a-1}^{(k)}.
\]

Use robot-specific \(H_a\) in the data pipeline:

- Panda: 4,
- UR5e: 6,
- Sawyer: 8,

and execute the complete chunk before the next high-level decision.

### 7.4 Shared latent action

Learn:

\[
u_t = E_\theta(z_t, a_{t:t+H_a-1}^{(k)}, e_k),
\qquad u_t \in \mathbb{R}^{d_u},
\]

with \(d_u=8\) in the first stage.

Decode with an embodiment-specific decoder:

\[
\hat a_{t:t+H_a-1}^{(k)} = D_{\phi_k}(u_t,e_k).
\]

Safety projection:

\[
\tilde a^{(k)} = \Pi_k(\hat a^{(k)})
\]

enforces joint limits, velocity limits, workspace constraints, gripper range, and chunk smoothness.

### 7.5 RL interface

The high-level policy outputs a latent action:

\[
u_t \sim \pi_\omega(u_t \mid z_t,l,e_k).
\]

The environment receives:

\[
a_t^{(k)} = \Pi_k(D_{\phi_k}(u_t,e_k))_0,
\]

or the full decoded chunk when the low-level controller supports chunk execution.

---

## 8. Method

### 8.1 Architecture

Proposed components:

1. **State/context encoder**
   - RGB-D encoder,
   - language/object context,
   - proprioception,
   - morphology descriptor.
2. **Action-chunk encoder**
   - native action sequence,
   - robot-specific padding/masking,
   - temporal positional encoding.
3. **Shared latent projection**
   - 8- or 16-dimensional action code.
4. **Robot-specific decoder**
   - independent per embodiment,
   - optional FiLM/adapter from morphology.
5. **Progress consequence head**
   - predicts task-progress change.
6. **Safety layer**
   - deterministic and differentiable where possible.
7. **High-level actor**
   - outputs latent action; decoder frozen during first-stage RL.

### 8.2 Objectives

Use the following notation:

\[
\mathcal{L}
= \mathcal{L}_{\text{rec}}
+ \lambda_{\text{align}}\mathcal{L}_{\text{align}}
+ \lambda_{\text{cycle}}\mathcal{L}_{\text{cycle}}
+ \lambda_{\text{prog}}\mathcal{L}_{\text{prog}}
+ \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}}.
\]

#### Reconstruction

Masked smooth-L1 or L1 in native action space:

\[
\mathcal{L}_{\text{rec}}
= \|M_k(\hat a^{(k)}-a^{(k)})\|_1.
\]

#### Cross-embodiment alignment

For paired phases from embodiments \(i,j\):

\[
\mathcal{L}_{\text{align}}
= \|u^{(i)}-u^{(j)}\|_2^2.
\]

Include hard negatives:

- same task but wrong phase,
- same phase but wrong object,
- wrong task,
- shuffled pair labels.

#### Cycle consistency

Decode the latent action from embodiment \(i\) through decoder \(j\), re-encode with embodiment \(j\), and match the original latent.

#### Progress consequence

\[
\mathcal{L}_{\text{prog}}
= \left\|\Delta p_i(u)-\Delta p_j(u)\right\|^2.
\]

Do not require identical low-level trajectories; require consistent task progress.

#### Temporal smoothness

Penalize abrupt latent changes within a trajectory, especially around contact and grasp phases.

### 8.3 Safety projection

Define explicit projection:

\[
\Pi_k(a)
= \operatorname{Clip}\left(a; l_k,u_k,\dot q_{\max},g_{\min},g_{\max},W_k\right).
\]

Report:

- pre-projection violation rate,
- post-projection violation rate,
- clipping magnitude,
- task impact of projection,
- chunk jerk.

### 8.4 Behavior cloning

Two strategies:

1. native-action BC,
2. latent-action BC with frozen decoder.

For latent BC, standardize target latents and optimize through the frozen decoder:

\[
\mathcal{L}_{\text{BC}}
= \|\hat u-u\|_1
+ \lambda_{\text{act}}\|D_k(\hat u)-a^{(k)}\|_1.
\]

This decoder-aware term is essential because a latent code has no unique physical meaning without its decoder.

### 8.5 Online RL

Use PPO in latent action space:

\[
\pi_\omega(u_t|z_t).
\]

Freeze:

- visual backbone,
- shared action encoder,
- robot-specific decoder.

Compare with native-action PPO under:

- identical environment steps,
- identical PPO hyperparameters,
- identical evaluation episodes,
- identical random seeds,
- identical task distribution.

### 8.6 New-embodiment adaptation

For an unseen robot:

1. freeze the shared action encoder and high-level actor,
2. train only a new decoder/adapter,
3. evaluate at 5, 20, and 50 demonstrations.

Separate two metrics:

- **teacher-latent decoder adaptation:** encode demonstration actions, train the decoder, and evaluate reconstruction/execution;
- **closed-loop policy adaptation:** feed policy-generated latents into the decoder and evaluate task success.

Do not use teacher-latent gains to imply closed-loop policy transfer.

---

## 9. Data and C0 Engineering Protocol

### 9.1 Minimal unified environment

Describe:

- MuJoCo,
- Panda, UR5e, Sawyer,
- pick, place, open-drawer, push,
- shared scene and initial-state construction,
- robot-specific action dimensions.

### 9.2 Data scale

Current C0 pilot:

- 3 embodiments,
- 4 tasks,
- 100 paired scenes per task,
- 1,200 total trajectories,
- train/validation/test = 70/15/15,
- 120 failure trajectories,
- 120 perturbation trajectories.

Final paper target:

- 3-5 embodiments,
- 4-6 tasks,
- 200-500 pairs per task after the pilot,
- clearly separated in-distribution and OOD test sets.

### 9.3 Observations

Every chunk records:

- 128×128 RGB,
- metric depth,
- camera intrinsics,
- camera extrinsics,
- EEF pose,
- joint positions/velocities,
- gripper state/action,
- native action chunk,
- task progress,
- success,
- contact count,
- safety flags.

### 9.4 C0 gates

Current verified gates:

| Gate | Result |
|---|---|
| Repeated reset/step | Passed for 12 robot-task combinations |
| RGB-D/EEP alignment | 0.66-0.83 px pixel error |
| Depth alignment | 0.018 m error |
| Post-safety boundary violations | 0 |
| Pair/task/scene leakage | 0 |
| Batch-8 GPU memory | 0.260 GB reserved |
| Chunk latency median | 0.64-0.87 ms |
| Nominal expert success | 100% for pick/place/open; 81-84% for push |

State that the current arm models are minimal kinematic MJCF assets, not vendor-grade dynamics. The final paper should replace them with Robosuite/Menagerie models before making claims about physical transfer.

---

## 10. Experimental Questions

### EQ1: Are semantic actions aligned across embodiments?

Metrics:

- exact-pair retrieval R@1/R@5,
- similar-action retrieval R@1/R@5,
- cycle consistency,
- task-progress consistency,
- latent effective rank,
- per-embodiment reconstruction error.

### EQ2: Do decoded actions remain executable?

Metrics:

- native reconstruction MAE,
- semantic action MAE,
- post-safety joint-limit violation rate,
- velocity violation rate,
- executable-chunk rate,
- jerk,
- inference latency.

### EQ3: Does the latent interface preserve single-robot control?

Metrics:

- BC action error,
- success rate,
- executable rate,
- collision rate,
- chunk smoothness.

### EQ4: Does CELAR improve online RL efficiency?

Primary metrics:

- final held-out success rate,
- environment steps to 50% success,
- environment steps to 80% success,
- area under learning curve.

Secondary metrics:

- latent entropy,
- collapse indicators,
- collision/limit rates,
- training stability.

### EQ5: Does the representation transfer to a new embodiment?

Metrics:

- 5/20/50-shot action error,
- executable rate,
- closed-loop success,
- negative migration,
- decoder parameter count,
- adaptation wall-clock time.

### EQ6: Does multi-task training preserve shared semantics?

Metrics:

- per-task success,
- task-to-task interference,
- latent cluster overlap,
- cross-task retrieval,
- task-conditioned progress accuracy.

### EQ7: Does the method generalize under OOD conditions?

Test:

- unseen objects and sizes,
- unseen initial poses,
- camera shift/lighting/background,
- language paraphrases,
- controller delay,
- gripper variation,
- contact disturbance.

---

## 11. Baselines

### Representation and action-interface baselines

1. Independent per-embodiment action autoencoder.
2. Shared raw action autoencoder with robot ID.
3. Normalized joint action space with robot ID.
4. End-effector delta plus IK.
5. Action VAE.
6. VQ-VAE action tokenizer.
7. Shared decoder without morphology conditioning.
8. CELAR without cross-embodiment alignment.
9. CELAR without cycle consistency.
10. CELAR without progress consequence.
11. CELAR with shuffled pair labels.

### Policy baselines

1. Per-embodiment native-action BC.
2. Shared native-action policy with embodiment ID.
3. Native-action PPO.
4. Normalized-joint shared PPO.
5. End-effector delta + IK PPO.
6. Action-VAE PPO.
7. Shared-decoder PPO.
8. CELAR PPO.

### Adaptation baselines

1. Independent BC trained from scratch on 5/20/50 demonstrations.
2. Independent action autoencoder trained from scratch.
3. Shared policy with a new robot-ID embedding.
4. CELAR teacher-latent decoder adaptation.
5. CELAR frozen-policy + decoder adapter in closed loop.

---

## 12. Ablations

Run:

1. latent dimension: 4, 8, 16, 32;
2. with/without morphology descriptor;
3. with/without paired alignment;
4. with/without cycle loss;
5. with/without progress-consequence loss;
6. with/without smoothness;
7. with/without safety projection;
8. frozen vs partially unfrozen decoder;
9. paired vs unpaired public data;
10. RGB vs RGB-D;
11. with/without language;
12. with/without proprioception;
13. action-chunk length 1, 4, 8;
14. shared latent vs shared decoder;
15. teacher-latent vs closed-loop adaptation.

---

## 13. Current Pilot Evidence and How It Maps to the Paper

### C1: Cross-embodiment alignment

Current proxy results:

- CELAR similar-action retrieval R@1: 0.911,
- shared autoencoder: 0.550,
- shuffled-pair control: 0.378,
- executable rate: 0.981,
- effective latent rank: 4.01 / 8.

Paper use:

- preliminary evidence for EQ1/EQ2,
- must be replicated on MuJoCo/Robosuite,
- do not combine proxy and final benchmark numbers in the same primary result table.

### C2: Single-embodiment behavior cloning

Current proxy results:

- native BC semantic MAE: 0.2553,
- CELAR latent BC semantic MAE: 0.2827,
- executable-rate ratio: 0.860,
- CELAR jerk: 0.0478 vs native 0.1007.

Paper framing:

- CELAR preserves a usable control interface rather than improving action precision,
- explain the modest semantic-error gap as the price of a compact shared bottleneck,
- emphasize lower jerk and shared interface benefits.

### C3: Equal-budget PPO

Current proxy results:

| Method | Final success | Steps to 50% |
|---|---:|---:|
| Native PPO | 0.375 | 42,325 |
| Shared-decoder PPO | 0.535 | 40,960 |
| CELAR PPO | 0.715 | 36,864 |

Paired improvement over native PPO: +34.0 percentage points, 95% CI [+22.9, +39.6], p=0.026.

Grouped audit:

- 18 seed-body-task pairs,
- mean improvement +29.4 points,
- 95% CI [+17.8, +40.5],
- 83.3% positive directions,
- 5/6 body-task combinations favor CELAR.

Paper use:

- central pilot evidence for EQ4,
- must be repeated against vendor-grade robot models,
- include all three seeds and per-task curves,
- avoid describing 12.9% fewer steps to 50% as a 30% efficiency result; the current gate passes primarily on the +5-point success criterion.

### C4: Unseen embodiment

Current proxy results:

- 5-shot CELAR decoder semantic MAE: 0.1110,
- independent action AE: 0.1500,
- 26.0% error reduction,
- 100% executable rate,
- 20-shot comparison remains positive.

Negative result:

- closed-loop frozen-policy adaptation remains worse than independent BC.

Paper framing:

- claim decoder-level few-shot adaptation,
- do not claim end-to-end policy transfer,
- discuss the gap as a central limitation and future-work direction.

### C0: MuJoCo data engineering

Current results:

- 1,200 paired trajectories,
- zero leakage,
- zero post-safety violations,
- 0.66-0.83 px RGB-D alignment,
- 0.018 m depth error,
- batch-8 memory 0.260 GB,
- chunk latency below 1 ms in the minimal model.

Paper use:

- describe as an implementation and data-quality contribution,
- explicitly identify the kinematic MJCF simplification,
- replace with vendor-grade models for the main submission if possible.

---

## 14. Planned Main Experiments

### Experiment 1: Paired action alignment

- Embodiments: Panda, UR5e, Sawyer.
- Tasks: pick, place, open-drawer, push.
- Split: disjoint task instances, scenes, pair IDs.
- Seeds: 3.
- Metrics: exact-pair R@1/5, similar-action R@1/5, cycle error, progress consistency.

### Experiment 2: Executable reconstruction

- Same data.
- Compare 11 representation methods.
- Report per-embodiment, not only pooled averages.
- Include safety-layer trigger and post-clipping error.

### Experiment 3: Single-robot BC

- Native BC, EEF BC, action-VAE BC, shared latent BC, CELAR latent BC.
- Evaluate in-distribution and unseen episodes.
- Include action error, success, executable rate, collision, jerk, latency.

### Experiment 4: Equal-budget known-body PPO

- Four tasks.
- Three seeds.
- Same environment-step budget.
- Primary result: final success and steps to target success.
- Include per-task and per-body curves.

### Experiment 5: Unseen-body adaptation

- Train on Panda and UR5e.
- Hold out Sawyer.
- Budgets: 5, 20, 50 demonstrations.
- Compare independent BC, independent AE, shared-ID policy, teacher-latent decoder, and closed-loop adapter.

### Experiment 6: Multi-task and OOD

- 4-6 tasks.
- Object, pose, camera, language, and delay shifts.
- Report positive migration and negative migration.

### Experiment 7: External benchmark

At least one of:

- LIBERO/LIBERO-Plus,
- RoboTwin 2.0,
- CALVIN,
- SimplerEnv.

The external benchmark should test whether the action-interface claim survives a different simulator and task suite.

---

## 15. Statistical Analysis

### Pairing

Pair by:

- random seed,
- embodiment,
- task,
- scene,
- evaluation episode.

### Uncertainty

Report:

- mean,
- standard deviation,
- 95% bootstrap confidence intervals,
- paired differences,
- Holm correction for multiple comparisons.

### Sample-efficiency curves

Plot:

- mean held-out success,
- interquartile or 95% confidence band,
- target thresholds at 50% and 80%.

### Required claims

Use "improves" only when:

- the mean difference exceeds the pre-registered threshold,
- the confidence interval excludes zero or the pre-registered practical threshold is met,
- the direction is consistent across most seeds/tasks.

Otherwise use:

- "trends toward,"
- "is associated with,"
- "has potential to,"
- "was not sufficient to establish."

---

## 16. Figures

### Figure 1: Problem and concept

Three robot arms executing the same task phase map to a shared latent neighborhood but decode through robot-specific controllers.

### Figure 2: CELAR architecture

RGB-D + language + proprioception + morphology → action encoder → shared latent → robot-specific decoder → safety projection → native chunk.

### Figure 3: Paired data pipeline

Shared scene and initial state → robot-specific trajectories → RGB-D, action chunk, EEF pose, grip, progress, safety.

### Figure 4: Cross-embodiment retrieval

2D/UMAP latent plot and R@1/R@5 bar chart for CELAR and baselines.

### Figure 5: Reconstruction and execution

Per-embodiment MAE, executable rate, jerk, and violation rates.

### Figure 6: Behavior cloning

Semantic error vs executable rate and smoothness.

### Figure 7: Online PPO sample efficiency

Learning curves by method and task; shaded confidence bands.

### Figure 8: New-embodiment adaptation

5/20/50-shot semantic error and closed-loop success.

### Figure 9: Ablation matrix

Alignment, cycle, outcome, smoothness, morphology, safety, latent dimension.

### Figure 10: Failure analysis

Collision, limit activation, grasp failure, task-phase confusion, latent collapse.

---

## 17. Tables

### Table 1: Embodiments

Robot, DoF, action convention, limits, gripper, control rate, camera layout.

### Table 2: Dataset splits

Task, scene, pair, embodiment, trajectory counts, failure, perturbation.

### Table 3: Representation results

Methods × retrieval/reconstruction/execution/cycle/progress.

### Table 4: Behavior cloning

Methods × semantic error/success/execution/jerk.

### Table 5: Online PPO

Methods × final success/steps-to-target/AUC/collision/latency.

### Table 6: Unseen embodiment

Shots × adaptation method × semantic error/execution/closed-loop success.

### Table 7: OOD and external benchmark

Shift type × success × interaction efficiency.

### Table 8: Resource usage

Peak memory, latency, parameters, training GPU-hours, storage.

---

## 18. Discussion

### Why the direction may matter

- It separates semantic sharing from physical execution.
- It allows different robot architectures to share the high-level policy while preserving native control.
- It provides a concrete mechanism for few-shot decoder adaptation.
- It can reduce data duplication when tasks are re-collected across robot families.

### What the evidence does not show yet

- Vendor-grade robot dynamics.
- Real-world camera noise and calibration drift.
- Real contact-rich manipulation.
- Reliable closed-loop policy transfer to unseen robots.
- Broad multi-task sharing.
- Safety under human interaction.

### Negative and mixed findings

- C2 has slightly higher semantic error than native BC.
- C4 teacher-latent adaptation is stronger than closed-loop policy adaptation.
- Push success is lower than pick/place/open in the C0 expert data.
- Many gains are measured in a controlled proxy and may shrink under realistic dynamics.

These findings should be presented rather than hidden because they define the claim boundary and motivate future work.

---

## 19. Limitations

1. Small number of embodiments and tasks.
2. Paired data require controlled resets that are not available in arbitrary real-world datasets.
3. Robot-specific decoders add parameters and deployment complexity.
4. The shared latent may discard task-relevant low-level details.
5. Safety projection may alter intended behavior near contact.
6. New-robot adaptation depends on the quality and coverage of demonstrations.
7. Cross-embodiment alignment assumes a meaningful shared task-phase definition.
8. Simulator and real-world action distributions differ.

---

## 20. Conclusion

Draft conclusion:

> We present CELAR, a cross-embodiment latent action framework that separates shared action semantics from robot-specific executable decoding. In controlled studies, CELAR aligns semantically similar actions across Panda, UR5e, and Sawyer, maintains a usable latent control interface, improves equal-budget PPO success, and supports few-shot decoder adaptation for a held-out embodiment. At the same time, our results expose an important gap: action-interface transfer does not automatically yield closed-loop policy transfer. These findings motivate a carefully scoped next stage centered on vendor-grade multi-embodiment simulation, robust latent priors, and closed-loop adaptation.

For the current pilot, add:

> The present quantitative results are from controlled proxy environments and should be treated as preliminary evidence rather than final robotics claims.

---

## 21. Reproducibility Checklist

Report:

- code commit,
- random seeds,
- data generation version,
- scene and task IDs,
- split hashes,
- action dimensions and joint order,
- safety thresholds,
- PPO hyperparameters,
- number of environment steps,
- GPU model and memory,
- inference hardware,
- latency measurement protocol,
- checkpoint hashes,
- complete failure logs.

Artifacts from the current pilot:

- `results/c0_mujoco_v1/`
- `results/celar_pilot_v1/`
- `results/celar_c2_v4/`
- `results/celar_value_v3/`
- `results/celar_v3_group_analysis/`
- `results/celar_combined_v1/`

---

## 22. Writing Order

1. Finalize problem formulation and notation.
2. Build the main MuJoCo/Robosuite result tables.
3. Decide which claims survive vendor-grade dynamics.
4. Freeze the primary and secondary metrics.
5. Write results before introduction.
6. Write limitations and failure taxonomy.
7. Write the method section only after the final architecture is fixed.
8. Write abstract and introduction last.
9. Run a claim-to-evidence audit: every number and causal statement must map to a table, figure, or appendix.

---

## 23. Claim-to-Evidence Audit

### Safe to claim now

- A paired data pipeline for three robot families and four manipulation tasks has been implemented.
- C0 data, RGB-D alignment, action safety, leakage, memory, and latency gates pass in the current environment.
- CELAR improves similar-action retrieval over a shared autoencoder in the controlled proxy.
- Latent BC can maintain a usable interface under the tested pilot threshold.
- CELAR PPO outperforms native-action PPO in the current proxy pilot.
- A held-out decoder can adapt from few demonstrations at the action-reconstruction level.

### Requires completed follow-up before claiming

- CELAR improves RL sample efficiency on vendor-grade MuJoCo/Robosuite robots.
- CELAR transfers to a new physical robot without retraining the high-level policy.
- CELAR is robust to OOD cameras, objects, language, and controller latency.
- CELAR improves real-world manipulation success.
- CELAR supports long-horizon or bimanual manipulation.

### Must not claim

- Zero-shot cross-robot deployment.
- Universal actions across all robot types.
- General real-world safety.
- A complete end-to-end CELAR method on the basis of proxy-only evidence.
- That action-interface adaptation guarantees task-success transfer.

