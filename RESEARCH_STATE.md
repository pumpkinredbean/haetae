# Haetae research state

Status: waiting

## Objective and completion criteria

Build and evaluate a clean-room local System One decision model that is structurally closer to Jev than an NLI wrapper: one packed non-generative forward pass per typed question; dynamic Choice, Noul, and Score outputs; calibrated probabilities; Korean support; and local serving through a Jev-compatible API.

Completion requires a reproducible completed checkpoint, named held-out evaluations, temperature calibration and selective-risk evidence, permutation and option-perturbation stress tests, local CPU and MPS latency, and a ChatGPT 6 Pro review of the final code and measured results.

## Repository and external review

- Repository: https://github.com/pumpkinredbean/haetae
- Branch: `main`
- Accepted execution-benchmark code commit: `73d2b49c3d95267b135adbcb3af5f53ede3426d3`
- Checkpoint format 3 implementation commit: `a30ba8b`
- Exact-review fix commit: `06b8728`
- Aside conversation: `Clean Room Jev Reproduction`
- Aside URL: https://chatgpt.com/c/6aaeba00-b670-83e8-9c29-3370b3c7945d
- Review model: ChatGPT 6 Pro
- Interaction method: Aside REPL only
- Exact review of commit `a30ba8b`: publication, rotation, AdamW checks, real signal continuation, and epoch rollover passed. Reported semantic-validation, tokenizer-identity, calibration-binding, descriptor-schema, completed-target, recovery-display, and certification-policy findings are fixed in the current working tree.
- Targeted re-review of exact commit `06b8728`: ChatGPT 6 Pro inspected the complete commit and ran additional malformed-RNG, scheduler, AdamW-counter, tokenizer-identity, exact-resume, stale-directory, descriptor, and completed-target probes. It reported 27 passes and one expected MPS skip in its CPU environment, closed both pre-run blockers, and explicitly approved starting the unattended 1,536-token run.

## Confirmed model design

- ModernBERT-base backbone.
- One row per question: `[CLS] state [SEP] instructions [SEP] option1 [SEP] option2 ...`.
- A shared scalar head reads the bidirectional hidden state of the separator preceding each option.
- Candidate logits are normalized only within their question.
- Choice order is augmented during training. Score order and Noul yes-first semantics remain fixed.
- Hard or soft cross-entropy plus Brier is the baseline objective. Cross-entropy plus ranked probability score for Score is the first planned loss ablation.
- The mixture contains 13 sources, including KLUE-YNAT and NSMC for Korean.
- Calibration claims must identify the dataset and deployment conditions. Selective action uses a one-sided binomial risk bound.

## Updated comparative evidence

- TypeSafe still has not published Jev's weights, parameter count, attention graph, base model, or RLCD recipe. The defensible target is its observed behavior rather than a claim of architectural reproduction.
- Archer Hume's 2026-09-17 black-box study provides evidence for a shared state computation, isolated question branches, listwise option interaction, and a typed parallel readout. The exact internal mechanism remains an inference.
- `jaredpalmer/kev` commit `20fa6268c8ceb226530be2fb5266ab2c36b37724` implements that reconstruction with a causal Qwen backbone, a block-causal branch mask, restarted branch positions, and a pointer head. It also publishes frozen development and test suites and direct Jev comparisons. This is the strongest inspected open structural reproduction so far.
- The active Haetae model is therefore a lightweight per-question encoder baseline, not the closest known multi-question reconstruction. Its remaining potential advantages are a much smaller local backbone, Korean training data, soft ordinal supervision, and digest-bound calibration and certification artifacts.
- The next architecture experiment should share one state across isolated question branches and add a listwise decision readout while preserving Choice permutation tests and Score order. Compare it on `kev`'s frozen transfer development suite and on a separately frozen Korean suite before reading any new locked test.

## Implemented review findings

- The optimizer covers every trainable backbone and head parameter exactly once, with no decay on one-dimensional backbone parameters.
- Candidate descriptions cannot be silently truncated. State tokens yield the budget first; records whose fixed question and candidates do not fit are explicitly rejected.
- Loss inputs, option counts, target shapes, label bounds, finiteness, and normalization are validated.
- Noul serving no longer splits `yes` into character candidates.
- Civil Comments hard-label orientation is correct.
- HelpSteer2 records sharing an exact prompt use the same parent key and cannot cross the train-validation split.
- Temperature is fitted in log space. Conformal rank and binomial boundary cases are validated.
- Evaluation and serving explicitly handle rejected packs.
- Multi-question serving batches all questions into one backbone call.
- Choice permutation includes binary choices and keeps labels and soft targets attached to their semantic option.

## Checkpoint format 3

The working tree replaces the path-based format 2 checkpoint with an immutable run and generation protocol:

- a nonblocking `fcntl` lock is held for the run directory for the process lifetime;
- `run.json` binds a UUID to training configuration, exact train and validation fingerprints, backend, optimizer parameter names/order/shapes, the complete serialized tokenizer pipeline and wrapper settings, model configuration, training-code contents, and software versions;
- a fresh run refuses to overwrite an existing run directory, while automatic resume requires the exact run identity;
- every generation uses a unique temporary file, file `fsync`, macOS `F_FULLFSYNC`, atomic rename, directory `fsync`, byte size, and SHA-256;
- `latest.json` atomically publishes the current generation and one previously validated generation; `progress.json` is derived display data;
- the serialized generation is re-opened and validated before manifest publication;
- automatic rollback is limited to a missing file, byte-size mismatch, or SHA-256 mismatch. Run identity, format, environment, NaN, and optimizer inconsistencies fail closed;
- the loader validates schema, integer ranges, model shapes and finite values, AdamW moment coverage/shapes/dtypes/finiteness/nonnegative second moments/step counters, scheduler continuity, shuffled row order, cursor, RNG state, and backend before mutating live objects;
- PyTorch places AdamW moments on the parameter device. The scalar step counter is not moved indiscriminately;
- lifecycle states are `running`, `interrupted`, `completed`, and `failed_no_progress`;
- serving and certification load only a completed manifest whose checkpoint reached the configured target, and verify tokenizer and model-configuration fingerprints;
- publication temporarily needs space for the two retained generations plus the new temporary generation. With ModernBERT this is about 5.1 GB.

Validation completed locally:

- twenty-six focused unit tests pass, including physical corruption fallback, semantic-corruption fail-closed behavior, malformed descriptor rejection, manifest-publication failure, process lock exclusion, rejected epoch-tail continuation, full RNG/scheduler restoration validation, tokenizer-pipeline identity, bound calibration artifacts, certification graph detachment, no-progress failure, calibration boundaries, and exact resume;
- the CPU exact-resume test matches record IDs, option permutations, Python and Torch random draws, losses, learning rates, optimizer state, data state, and final parameters against an uninterrupted run;
- the native MPS test confirms restored moments are on MPS, AdamW scalar steps stay on CPU, and the next update matches the uninterrupted branch exactly;
- the post-review real ModernBERT/MPS smoke run used run ID `ae2f88a4-be47-4190-9680-2ce90b87c716`, saved interrupted generation 4 at step 2, loaded the persisted tokenizer and resumed that exact generation, then completed generation 10 at step 8;
- completed generation 10 has SHA-256 `d56ec0f1bdc6fde37e15a2d89a8e1fe4d0bd860771705bafa50379e5acac5be5` and passed the completed serving loader with `calibrated: false` and the recorded context limit.

The smoke run validates mechanics only. It is not a model-quality result.

## Historical training artifacts

- `runs/v1` is a debugging artifact. Its `model.pt` predates a lost long run and is not a valid result.
- `runs/v2` is checkpoint format 2. It proved exact cursor recovery from step 82 and cursor 656 to step 100 and cursor 800. It was stopped at a valid periodic step-250 checkpoint before format 3 work and will not be migrated into the final baseline.
- `runs/smoke-v3d` is the post-review completed SIGTERM and resume test. Earlier mechanics artifacts were removed after their evidence was superseded.

## State truncation audit

The actual tokenizer and packing policy were audited on 1,700 rows per ordinary source and 1,700 HelpSteer2 source rows expanded to 8,500 questions.

- At 768 tokens, 11 sources had no state truncation.
- BoolQ truncated 1 of 1,700 states, retaining 82.0% of that state.
- HelpSteer2 truncated 765 of 8,500 questions, or 9.0%. Mean retained state was 98.0%, but the worst retained state was 34.4%.
- HelpSteer2 truncation fell to 3.71% at 1,024 tokens, 0.29% at 1,536, and 0.06% at 2,048.
- A worst-case batch of the eight longest HelpSteer2 questions completed a 1,536-token MPS forward and backward pass in 7.38 seconds, with 2.23 GiB current and 32.30 GiB driver allocation after the optimizer step.
- The same eight-example batch at 2,048 tokens failed during forward attention at 46.53 GiB allocated, while requesting another 1.50 GiB against the 47.74 GiB MPS limit.

The planned clean baseline uses 1,536 tokens. This sharply reduces HelpSteer2 response loss while keeping the configured batch size viable; 2,048 tokens is not viable for the measured worst-case batch.

## Resource-limited baseline attempt

The reviewed baseline started from a fresh directory at 2026-09-20 14:15:56 KST. It uses MPS and the exact command below:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3 "zsh -lc 'set -o pipefail; HF_HUB_OFFLINE=1 uv run python -u -m haetae.train --sources ag_news,banking77,massive,mnli,anli,arc,emotion,klue_ynat,boolq,nsmc,civil_toxicity,sst5,helpsteer2 --per-source 1500 --eval-per-source 200 --steps 3000 --batch 8 --max-len 1536 --out runs/v3 --save-every 50 --resume none 2>&1 | tee -a train_v3.log'"
```

The attempt is retained for checkpoint and resource evidence. Do not resume it as the final baseline:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3 "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m haetae.train --sources ag_news,banking77,massive,mnli,anli,arc,emotion,klue_ynat,boolq,nsmc,civil_toxicity,sst5,helpsteer2 --per-source 1500 --eval-per-source 200 --steps 3000 --batch 8 --max-len 1536 --out runs/v3 --save-every 50 --resume auto 2>&1 | tee -a train_v3.log'"
```

- tmux session: `haetae-v3`
- log: `/Users/minkyu/workspace/haetae/train_v3.log`
- authoritative run identity: `/Users/minkyu/workspace/haetae/runs/v3/run.json`
- authoritative generation manifest: `/Users/minkyu/workspace/haetae/runs/v3/latest.json`
- derived progress: `/Users/minkyu/workspace/haetae/runs/v3/progress.json`
- run ID: `05006c2d-ae6b-4821-8c6c-d4ab27b43bfc`
- run-spec SHA-256: `60c551c578de6446ae60113009403bb5091b0391ee549bcd9fc50d289123014f`
- training-data SHA-256: `372dfb7edbe90dbf61be1ab8171a8833f74eba616b39103ecee9896e184be5c1`
- validation-data SHA-256: `c95c9c32397584182f1e75ca6accfbbe256b4cee2b2c1698fe0d15242b78758b`
- tokenizer SHA-256: `e18014b047f21133e8cc5029313c14f5d90beee97eb19f307d5f37af8631178e`
- model-configuration SHA-256: `d3e57da889eaf2bdf87744ea4c663159a36f37f46c182c418dbfbd1051c38aec`
- actual data: 25,500 training questions from 18,750 parents and 3,400 validation questions from 2,500 parents;
- first durable training generation: generation 2, step 50, SHA-256 `122d823b442ac0a2864be6472bce700d1f743eb003fb7ca9ee8a3b6d74493953`;
- step 50 loss: 1.3141, with zero rejected batches and 2.17 seconds per step including initialization and checkpoint publication.
- The first process reached 43 GiB of unified MPS memory while finishing update 97. SIGTERM completed the active backward pass and published interrupted generation 3 at step 97 with SHA-256 `044f50df0578e239f88b6966d8c20d06ebf58acf0953e0ec2c10d5987102b79c`.
- The process resumed from that exact generation with an MPS soft collection threshold of 0.9 and hard allocation limit of 1.3 times the 28.08 GiB recommended working set. Generation 4 reached step 100 with SHA-256 `11d28983d09d8aafb7e193e5e938368721840f5168aef4f3541217bbc006e77e`.
- The next long batch required more than the 36.50 GiB hard limit and raised an explicit MPS out-of-memory error. A final controlled resume and SIGTERM published interrupted generation 5 at step 101 with SHA-256 `e714f96a5af0443a00a339521fe520bea8782bcfbb566031e9d9e8c914141dba`.

## Microbatch replacement

The replacement recipe keeps an effective batch of eight questions but runs two four-question forward and backward passes before one clipped optimizer update. The run identity now records both effective batch and microbatch size.

- A regression test compares batch 2 against two microbatches of 1 and matches final model parameters within `rtol=1e-6`, `atol=1e-7`, with identical scheduler state.
- All 27 checkpoint, calibration, serving, and certification tests pass, including native MPS restore and the new accumulation test.
- A real ModernBERT MPS probe selected the eight longest HelpSteer2 questions from the measured pool and processed them as two microbatches of four at 1,536 tokens. The two passes took 3.165 and 2.912 seconds, peaked at 18.03 GiB driver allocation, and completed the AdamW update in 6.526 seconds. After cache release the driver allocation was 9.03 GiB.
- A real entry-point smoke run completed two MPS optimizer updates with effective batch 8 and microbatch 4. Run `7b958056-9fd0-4536-93e4-1ccea017daaf` published completed generation 3 with SHA-256 `f8ebdb47926e8b4eb36f9c9c96f3f7a4ac1822f1fefdd8a8a5ce2d5d2bafd5ae`; the completed serving loader answered a three-question Korean request and explicitly reported `calibrated: false`.
- ChatGPT 6 Pro reviewed exact commit `92e709aa4c6bef2a4d1664dd0fd7301c9235fd59` through the Aside REPL and returned `START — use fresh runs/v3-micro4`. Its independent checks covered uneven microbatches, rejected records, one optimizer and scheduler update per effective batch, real SIGTERM and SIGINT recovery, fresh-process exact resume, abrupt termination without partial publication, and microbatch identity mismatch rejection.
- The review found one nonblocking memory defect: the final quick validation still used the effective batch of eight. It now uses the immutable microbatch of four, and the training entry-point test verifies that argument.
- Start the replacement only in a fresh `runs/v3-micro4` directory. Do not resume `runs/v3` as the final baseline.

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3m4 "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m haetae.train --sources ag_news,banking77,massive,mnli,anli,arc,emotion,klue_ynat,boolq,nsmc,civil_toxicity,sst5,helpsteer2 --per-source 1500 --eval-per-source 200 --steps 3000 --batch 8 --microbatch 4 --max-len 1536 --out runs/v3-micro4 --save-every 50 --resume none 2>&1 | tee -a train_v3_micro4.log'"
```

The replacement run started from clean commit `cfaa7517f26660365259f22f5813a2fc1acfb85c` at 2026-09-20 15:23:41 KST.

- tmux session: `haetae-v3m4`
- run ID: `6c0474c2-7ed5-4513-9920-acbd1171e290`
- run-spec SHA-256: `9e5e500f998cdffa5c534dfa432e815796d19bfbb7067316d7b36c545c86d6b2`
- producer SHA-256: `f4dd1b79b0eaf1ee5e6b2cf8d1270cccf0f0f33b1040da4c9bf8b88ebbe7a216`
- immutable train and internal-validation fingerprints match the prior full attempt: `372dfb7edbe90dbf61be1ab8171a8833f74eba616b39103ecee9896e184be5c1` and `c95c9c32397584182f1e75ca6accfbbe256b4cee2b2c1698fe0d15242b78758b`;
- generation 1 is the validated running checkpoint at step 0;
- generation 2 is the first durable training checkpoint at step 50, SHA-256 `53fdd30ff443205a73dc62f0dc095f5800ac7148b3b916085350fba24024cd44`; step 50 loss was 1.3054 with zero rejected batches, and the run continued past step 75;
- The original tmux server disappeared after logging step 1225 without publishing an interrupt checkpoint. No writer or advisory lock remained, and no checkpoint or disk error was present. Automatic resume validated generation 25 at step 1200 and replayed step 1225 with the same logged loss of 1.1604. A second externally terminated tmux process stopped immediately after publishing generation 46 at step 2250. Automatic resume validated that generation and continued the same run. The later producer digest changed because measurement-only source files changed, while the immutable training-code digest and run specification remained exact.
- Generation 61 completed step 3000 with zero skipped batches. Its checkpoint SHA-256 is `838bd9802d89aa87599e79876d5cc3737f2f08406d83055706bdd080ba1f2606`, producer SHA-256 is `982c143ce66cee5a54f4779913aebb83fa55d4cfec4716257720df16517a3a14`, and the manifest status is `completed`.
- The post-completion internal validation scored accuracy `0.597` and unscaled ECE `0.030` over 3,400 questions in 120.6 seconds. KLUE-YNAT was `0.120` and NSMC `0.445`; this is an internal diagnostic, not the frozen external result.
- the final evaluation population was frozen immediately after start in `evaluations/v3-micro4`. Plan `4e08d1c39757ddcb9f9fff292634fcd28f2ba3c22cecdfa8d8d5a4d143ae66a2` contains A/B/C record counts 7,973/7,974/15,995 and parent counts 5,147/5,149/10,281. No model result was inspected before this assignment.

## Frozen measurement protocol

`haetae.measure` implements the three-role protocol requested by the external review.

- Each source has an explicit labeled evaluation-track registry. Hugging Face sources load at a declared commit revision; local Korean files are bound by SHA-256.
- Preparation reconstructs and verifies the run's consumed training and internal-validation fingerprints. It excludes complete parents with any exact normalized-record or state overlap.
- A, B, and C assignments are deterministic, source-order independent, and disjoint by record, state, and parent. A fits one source-balanced soft-target temperature. B freezes conformal thresholds and a Bonferroni-corrected selective-risk family. Only the explicitly gated final command reads C.
- The final artifact reports stable soft-target NLL, histogram and annotation Brier scores, full-class macro F1, calibration bins, Noul yes-probability reliability, ordinal RPS, expected-index error and threshold reliability, parent-cluster bootstrap intervals, paired raw-versus-scaled differences, uniform and training-only empirical-prior baselines, conformal coverage and normalized set sizes, and simultaneous selective-risk bounds. Zero accepted examples are recorded as not certified.
- Choice stress uses one formal question per parent and cell across all eligible sources. Raw and temperature-scaled permutation and controlled option-removal metrics reuse the exact retained state and instruction token prefix. Removal never deletes the gold option or the original top prediction, and binary questions are reported as ineligible rather than divided by zero.
- Latency freezes Noul, Score, highest-option-count Choice, longest-context, and digest-selected mixed workload entries. It records model-only and packing-inclusive raw timings and distributions for batch one and the evaluation batch. The default protocol uses 30 warmups and 200 timed trials in each of three independent runs. MPS trials synchronize before and after timing and record workload shape, software, hardware, dtype, attention implementation, and accelerator memory.

ChatGPT 6 Pro reviewed exact commit `cfaa7517f26660365259f22f5813a2fc1acfb85c` through the Aside REPL. It ran the 14 checked-in measurement tests plus 24 controlled probes and returned `MEASUREMENT HOLD; keep runs/v3-micro4 training unchanged`. The review reproduced a conformal floating-point boundary failure, unfrozen C inference settings, missing statistical files in the evaluator identity, incomplete consumed-parent checks, tie-obscured Choice action changes, and unrecoverable C artifacts after a latency failure. It also requested clearer soft-target reliability, ontology-safe macro-F1, and same-process latency terminology.

Commit `ee3d22a12d379c65b6686a22aa053a03d9eb55cd` fixes those findings:

- conformal prediction sets compare `1 - p <= q`, the exact score used during fitting;
- formal C scoring must match the frozen device, batch, dtype, attention implementation, software, tokenizer, model configuration, platform, and machine identity before C is read;
- code identity includes calibration, certification, checkpoint, data, evaluation, model, training, and measurement modules;
- consumed and cross-track parents use source-namespaced identities;
- permutation stress records the semantic option actually selected after position-based tie breaking;
- C logits, Choice stress, and latency are independently digest-bound and resumable;
- reliability distinguishes majority or consensus correctness from annotation-target probability;
- macro-F1 is unavailable unless every record shares one ordered label ontology;
- latency repetitions are labeled same-process repeated blocks.

The metadata-only parent audit reconstructed 19,945 consumed parent keys and inspected 20,577 parents in the original A/B/C files. Consumed overlap and cross-role overlap were both zero. The original role files therefore remain unchanged: A `d455efa9d75cf5833bc5be090cc462001fd45b4f7bf10e663a3c476fda106600`, B `b916f9ddef595c859af3beb569a6b5e1ff06b4b04bc96a48b442a14d38cdf0c5`, and C `e6ddd1277da8b061e0d4e4f76313bb2599efe1f1018676e55e1757073412b5b5`. Protocol amendment `0223911155398c36061fec96a62af8035569aa0561f71437cfe80f73e3947270` binds corrected code from commit `ee3d22a` to base plan `4e08d1c39757ddcb9f9fff292634fcd28f2ba3c22cecdfa8d8d5a4d143ae66a2` without rewriting membership.

ChatGPT 6 Pro returned `MEASUREMENT ACCEPT` for `ee3d22a` and `11a1c62`. Its controlled probes confirmed inclusive conformal ties, frozen-runtime rejection before synthetic C inference, one-time reusable prediction artifacts, separate annotation-target and hard-consensus reliability, parent isolation, semantic tie actions, ontology-safe macro-F1, and Unicode JSONL handling. It did not open real role C or independently verify the production overlap counts.

The accepted role A/B fitting job ran from 2026-09-20 17:36 KST to 17:42 KST. It did not deserialize role C or run C inference.

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3m4-fit "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m haetae.measure fit --plan evaluations/v3-micro4 --run-dir runs/v3-micro4 --device mps --batch 4 2>&1 | tee -a measure_v3_micro4_fit.log'"
```

- policy SHA-256: `0a5e0b2a9c271599610cdd6bd1cb8f0f597eef638ac27ec89924130be4f78fb8`
- fitted temperature: `1.6589349227635701`
- role A predictions: 7,973 accepted of 7,973, SHA-256 `583c48ab09fa8d6e800b8e30e0c2b633b0f8ae041ef1b5eb176bb5163e9a14b2`
- role B predictions: 7,974 accepted of 7,974, SHA-256 `be1fa782eb9fa92df61eb9585cb39c74d5015adc7e42af26d143a19daa5aa0c0`
- the frozen policy contains 18 formal cells and 54 simultaneous selective-risk bounds;
- calibration artifact SHA-256: `e3e2edce6d18024e4de2f8a2e5a32602225a2a4f030874572ead54a4556cb816`
- independent validation recomputed the policy and file digests, checked all logits and checkpoint bindings, and confirmed that no C prediction, stress, latency, or metric output existed.

The explicitly gated role C evaluation ran from 2026-09-20 17:46 KST to 18:09 KST. Its prediction, stress, and latency stages published separate digest-bound artifacts before the final report.

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3m4-eval "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m haetae.measure evaluate --plan evaluations/v3-micro4 --run-dir runs/v3-micro4 --device mps --batch 4 --allow-certification 2>&1 | tee -a measure_v3_micro4_eval.log'"
```

- tmux session: `haetae-v3m4-eval`
- log: `/Users/minkyu/workspace/haetae/measure_v3_micro4_eval.log`
- role C predictions: 15,995 accepted of 15,995, SHA-256 `53d925946ad65535233a7409581a347d559d5495ad17c44b2cda30bf11943d8b`, snapshot SHA-256 `e7f44c6e1496d544a0df137a3f7daff0523994002d9e9aa4d44a446ab685cfa3`;
- choice-stress artifact SHA-256: `f00898f3e63b58f1f2fddd8d7dffc01208cafc8910a6576a5b690f341bf01abc`;
- latency artifact SHA-256: `a143f44be3f19877fba0b004bffadbf655506e7182b94dfe82631e0621d5fcc6`;
- final metrics SHA-256: `eec509745ba2877cc5c66b55d7972f91af28fb68b8117a0064656c2a09f3406a`, file SHA-256 `c11cad1d0ad4beaa166df7e31d326397be695408d5b348e4e39dd58cc9d980e3`.

The frozen C result has overall accuracy `0.4903` with parent-bootstrap 95% interval `[0.4809, 0.4998]` and source-macro accuracy `0.5319`. Temperature scaling leaves accuracy unchanged, lowers NLL from `1.1935` to `1.1459`, and lowers histogram and annotation Brier scores by `0.00438`; all three paired parent-bootstrap intervals favor scaling. Source accuracies range from KLUE-YNAT `0.2387` and NSMC `0.4887` to AG News `0.8363` and Civil Toxicity `0.9217`.

Choice permutation preserves the semantic top action in `65.24%` of 20,460 variants with mean total-variation distance `0.0716`; controlled option removal preserves it in `89.24%` of 6,820 variants with mean total variation `0.0260`. The 90% conformal sets have observed per-cell coverage from `0.8588` to `0.9463`, often by returning large sets. None of the 54 simultaneous selective-risk rules certifies an error upper bound below 10%; the best bound is `0.1127` for Civil Toxicity. On the heterogeneous 64-record MPS workload, batch-one model latency averages `25.84 ms` per record and packing-inclusive latency `26.26 ms`; batch four averages `43.75 ms` and `44.46 ms` per record because dynamic padding includes long and many-option records.

All 70 repository tests pass. A HelpSteer2 mechanics run exercised A/B/C, five response-attribute cells, forced full conformal sets, selective abstention, and synchronized MPS latency. An AG News mechanics run exercised raw and scaled permutation, controlled removal, baselines, paired metrics, and latency. These are mechanics smokes, not quality results.

An offline preflight loaded every declared remote source at its exact cached commit plus both local Korean files, reconstructed the old full run's 25,500 training and 3,400 internal-validation questions, and froze every registered track without a role leak. At the default 1,600-parent cap per track, plan `e9440dafb69a756a5b8beab3cbc7fbcc72bc13af65a4f99690d05016e0478faf` contains A/B/C record counts 7,973/7,974/15,995 and parent counts 5,147/5,149/10,281.

## Shared-state experiment

- The prototype uses `jhu-clsp/mmBERT-small` at revision `abc32620dd4f6ab06f5fbe905dc25f310618e09f`, with four unforgeable decision delimiters and a pointer head.
- The state is encoded once. Each bidirectional question branch attends to the state and itself, while the state cannot attend to branches and sibling branches cannot attend to each other. Branch position IDs restart after the state.
- Tiny-model tests prove sibling isolation and packed-versus-separate equivalence. After correcting the ModernBERT half-window policy, a real untrained mmBERT CPU probe packed two Korean questions into 118 tokens instead of 62 and 76 separate tokens; maximum logit differences were `0` and `6.68e-6`. The pinned model has `local_attention=128` and a native radius of 64.
- The current Kev decision-v7 manifest SHA-256 is `a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2`; its verified training split SHA-256 is `7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`.
- Kev v7 has 12,576 training requests and 15,576 questions. Two thousand requests contain multiple questions. Every request fits the mmBERT 2,048-token policy without state truncation; the maximum is 1,011 tokens. Shared packing reduces unpadded training tokens by 9.44%.
- The transfer-v4 development split SHA-256 is `ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e`. Its 764 requests contain one question each, so it measures transfer quality but not multi-question efficiency.
- The trainer binds suite bytes, model revision, tokenizer, code, optimizer, scheduler, data order, and random state to immutable periodic generations. It averages question losses within a request and then averages requests, uses Choice permutation augmentation, and adds ranked probability score for Score questions.
- The evaluator fits one source-balanced temperature on the combined calibration partition and reports raw and scaled decision-v7, transfer-v4, and Korean development metrics. Macro-F1 is reported only within one fixed ordered option ontology.
- ChatGPT 6 Pro returned `SHARED EXPERIMENT HOLD` for commit `080d451` after reproducing two blockers: the custom mask used `local_attention` as a radius instead of `config.sliding_window`, and evaluation converted logits to float64 on MPS before moving them to CPU. Commit `2485f24d4d6aecda857a0c2927f0723b326e13c8` corrects both, adds the frozen v7 and Korean data path, and passes 70 tests plus a native-MPS conversion probe.
- ChatGPT 6 Pro then reviewed exact commit `2485f24` from GitHub and returned `SHARED EXPERIMENT START` for one development-only MPS seed. Its 28 controlled checks closed both prior blockers and exercised the public fetcher, combined freezer, Korean conversion, required development paths, request-balanced loss, remapping, and exact-resume mechanics. It identified two nonblocking audit restrictions: the freezer trusted supplied state digests instead of hashing rendered model input, and macro-F1 accepted matching keys without proving matching label meaning.
- Commit `0fad93144ba841541e50726ab982f0322189f366` closes both restrictions. The freezer now hashes the state produced by the runtime adapter, macro-F1 requires identical source, primitive, ordered keys, and ordered descriptions, and shared training requires a digest-bound rendered-state audit.
- ChatGPT 6 Pro reviewed that exact commit and again returned `SHARED EXPERIMENT START`. Its 23 controlled probes verified the two corrections, audit rejection paths, result-blind filtering, and both evaluator integrations. It found one remaining comparison-only binding gap: a comparison plan could supply transfer or Korean suites different from those covered by the rendered-state audit. Commit `aafd461cd9a43901a51a21898650365bea06a4a8` rejects that mismatch both when freezing and when loading a plan. The actual frozen transfer and Korean bindings equal the audited bindings.
- ChatGPT 6 Pro completed a narrow review of exact commit `aafd461cd9a43901a51a21898650365bea06a4a8` and returned `SHARED EXPERIMENT START`. It ran the five checked-in comparison-protocol tests plus 15 controlled probes. Substituted transfer or Korean suites, changed split digests or counts, missing split descriptors, and altered audit bytes were rejected before record access. It reported no additional training blocker and did not open either locked test.

The Korean comparison suite was frozen before any candidate-model result was inspected:

- `evaluations/korean-v1` excludes 41,891 state identities consumed by the baseline train, internal validation, and frozen A/B/C populations;
- KLUE-YNAT contributes 1,000 development and 1,000 locked parents, balanced across seven topic labels; NSMC contributes 1,000 and 1,000, balanced across positive and negative sentiment;
- 638 duplicate NSMC records were detected, including 39 exact texts with conflicting labels; conflicting states were excluded before selection;
- development and locked parents are disjoint, and the locked file remains gated and has not been parsed after freezing;
- the manifest file SHA-256 is `614a7c0a2febc617f2aaf34bf1906cc8bf0ed958e9138052a37da5ba5b04d5c1`; its canonical content SHA-256 is `b1f6fd98bc8b3efae9f1efaa4198bed3bb0e376d89d56df4be063b740cfedd4d`;
- the development split SHA-256 is `82db10f63886cd26f23eac79b0e5c5b87236d7b9eaf7102af5956fd3bdc2355b`. All 2,000 requests and 5,000 questions fit mmBERT without state truncation, with a maximum packed length of 242 tokens;
- shared-state packing uses 354,379 input tokens versus 429,873 for separate questions, a 17.56% reduction before padding. This is a token-count result, not a latency claim.

The combined training suite is frozen at `evaluations/shared-v1`:

- it contains the exact Kev v7 public partitions plus 2,996 Korean training requests and 400 Korean calibration requests reconstructed from the immutable baseline membership;
- three duplicate NSMC training records were found; one exact state had conflicting labels and also crossed into internal validation, so that state was excluded and same-label duplicates were collapsed;
- train has 15,572 requests and 23,068 questions, including 4,996 multi-question requests; calibration has 1,368 requests and 2,148 questions;
- all records fit mmBERT without state truncation, with maximum packed length 1,011; shared training packing reduces unpadded tokens by 11.02%;
- train, calibration, decision development, transfer development, and Korean development have zero exact-state or source-parent overlap across 20,084 states and 18,412 source-parent identities;
- the manifest file SHA-256 is `89b02fe5e76bdfae6ca5930d1e9b15b31dd61188330a68454cc951ac26c24d9c`; canonical content SHA-256 is `86dfb050f9a06483dada99e577e64a931f1dce074cfd2d14055972b35a05cded`;
- train, calibration, and decision-development SHA-256 values are `e6170e0f5b1920ca56f74a56f341d7fdf3472d70bbb8fe73ab72eb19dfacf22f`, `2b70af1e55714878d5377bb09ececcddf6481bcaf0e26d4ce3304d348fa76476`, and `1af33ba7170aa4ec64ccd83b81974a5c3c61071b7e5b9b2322d4565f05e36556`;
- the pinned fetcher deliberately materializes no Kev locked-test file. Neither Kev nor Korean locked test has been opened.

An exact-state audit against the baseline's immutable train and internal-validation membership found five overlapping requests in the public Kev decision-v7 development split, all from BoolQ. The transfer-v4 and Korean development splits have zero overlap. Model comparison will therefore report both the complete public development result and a common clean subset that excludes those same five requests from both models; the shared training suite and evaluation membership remain frozen.

Commit `720b62c` added that result-blind comparison, and commits `0fad931` and `aafd461` bind it to the corrected rendered-state audit and its exact external suites. Comparison plan `dc11cd8099fd3fe202145b20f4fee241ef98223b59ea3c16d1d243a3d73edd87` binds the completed baseline, all three public development suites, evaluator code, and exact exclusions. It removes 31 baseline-training-overlap requests containing 32 questions from the shared calibration partition for both models, while retaining 401 states from the baseline's held internal-validation partition as legitimate calibration data. It removes the five BoolQ development requests from both models' common-clean metrics. Baseline packing preflight accepted all 2,148 calibration and 7,232 development questions without state truncation.

Commit `4765215` added paired parent-bootstrap comparison of the two development reports. ChatGPT 6 Pro returned `COMPARISON ANALYSIS HOLD` after 49 controlled probes: question-task source names split parents that emit multiple primitives, both reports could agree on provenance that differed from the frozen plan, and a negative label indexed the final option. Commit `3e6d31d` closed those findings, but a second review found that retrying draws without every task source silently conditioned the bootstrap distribution. Commit `9cf90e30224ed6faacf12b08614fb12189c22bcd` uses fixed origin-source and task-source-incidence strata, retains every draw, fails unestimable singleton strata, and cross-checks the completed manifest and run-specification identity. ChatGPT 6 Pro returned `COMPARISON ANALYSIS ACCEPT` after 10 checked-in tests and 48 additional controlled probes. The accepted intervals are pointwise and conditional on fixed trained models, fitted temperatures, and observed stratum composition.

The corrected public-only rendered-state audit preserved the existing frozen suite bytes and again found zero overlap across 20,084 model-visible states and 18,412 source-parent identities. Its artifact SHA-256 is `b1d4c966d6a29493a3580f2f8e8b75ddf446f660f6e37e4a792e2b33ca487dbc`, and its file SHA-256 is `fa1300b3ec30ad321bacaa21ec41fcccb658b8b93378a75aef5d22aa739280b3`. All 79 repository tests pass.

A real pinned-mmBERT MPS probe completed one effective batch of eight as four microbatches of two. It deliberately paired the six longest 994–1,011-token, 77-option requests and the two longest multi-question requests. The optimizer update took `2.61` seconds; maximum observed driver allocation was `4.38 GB`, `14.52%` of the `30.15 GB` recommended maximum. A real entry-point smoke run then published an interrupt at step 4, restored generation 6 in a fresh process, continued through step 8 with the exact cursor and random state, and published interrupted generation 11. Smoke run ID is `91b2b05b-d116-41b6-a40f-412c76d36020`; generation 11 SHA-256 is `73c14456db412f3e2683b50a00b31b30fa05450eeb6ff73d4c6ef811db4d78d7`.

The fresh shared-state run is scheduled from 2026-09-20 18:18 KST. It performs exactly two request epochs: `ceil(15,572 / 8) * 2 = 3,894` optimizer updates.

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-shared-v1 "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m experiments.train_shared --suite evaluations/shared-v1 --steps 3894 --batch 8 --microbatch 2 --max-len 2048 --device mps --local-files-only --out runs/shared-v1 --save-every 50 --log-every 25 --resume none 2>&1 | tee -a train_shared_v1.log'"
```

Exact resume command:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-shared-v1 "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m experiments.train_shared --suite evaluations/shared-v1 --steps 3894 --batch 8 --microbatch 2 --max-len 2048 --device mps --local-files-only --out runs/shared-v1 --save-every 50 --log-every 25 --resume auto 2>&1 | tee -a train_shared_v1.log'"
```

- tmux session: `haetae-shared-v1`
- log: `/Users/minkyu/workspace/haetae/train_shared_v1.log`
- run ID: `71d06d48-a3cf-4ddc-bfe2-2315dbe66da8`
- run-spec SHA-256: `2797fc5a985c315f035658790c6ba582d634fe2ceeaa2d8fc5d35e78a100e9d5`
- training-code SHA-256: `ff54ea26399fd9ad335eabba9d6f6a85a49880d8ac23b5217de665b10c2cc3fe`
- generation 1 is the validated running checkpoint at step 0;
- generation 2 is the first durable trained checkpoint at step 50, SHA-256 `2a53635a19f109fdd2ed128aaac373a6186255266f7920eef52f013b71fef40b`;
- generation 37 is the latest validated checkpoint at step 1,800, SHA-256 `642c386555c9de2eec522768ca1bb43703fd8523bba04a51902f785ab38f6d19`;
- the original tmux server disappeared immediately after generation 37 with no writer, held lock, checkpoint error, or disk error. Automatic resume validated generation 37 and continued from the exact cursor and random state; step 1,825 logged loss `0.7845` in the resumed process.
- the resumed tmux server later disappeared immediately after generation 71 at step 3,500. Again there was no writer, held lock, checkpoint error, or disk error. Automatic resume validated generation 71 and continued at step 3,525.
- generation 79 completed the run at step 3,894 with SHA-256 `e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c`, epoch 1, cursor 15,572, and zero skipped batches. The completed loader revalidated the checkpoint body, run specification, manifest identity, target step, optimizer state, and digests.

## Development evaluation

The baseline evaluation is next and uses only frozen calibration and development files:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-baseline-dev "zsh -lc 'set -o pipefail; HF_HUB_OFFLINE=1 uv run python -u -m experiments.evaluate_baseline --run runs/v3-micro4 --comparison-plan evaluations/comparison-v1/plan.json --decision-suite evaluations/shared-v1 --transfer-suite evaluations/kev-transfer-v4 --korean-suite evaluations/korean-v1 --device mps --batch 4 --out evaluations/baseline-development.json 2>&1 | tee -a evaluate_baseline_development.log'"
```

The baseline evaluation completed with report SHA-256 `ed00a9a69420d478f7b751bcd69bc8171ffcca863f77175443d024a0abac2313` and file SHA-256 `20fd43e8db495bef1aadf6c3f73977d58042686d6f87e32a1c3e221234aa648b`. Its fitted temperature is `1.50927104494036`. Common-clean source-macro accuracy is `0.4992` on decision development, `0.5097` on transfer development, and `0.4206` on Korean development.

The shared-state evaluation uses the same frozen membership:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-shared-dev "zsh -lc 'set -o pipefail; HF_HUB_OFFLINE=1 uv run python -u -m experiments.evaluate_shared --run runs/shared-v1 --comparison-plan evaluations/comparison-v1/plan.json --decision-suite evaluations/shared-v1 --transfer-suite evaluations/kev-transfer-v4 --korean-suite evaluations/korean-v1 --device mps --batch 2 --out evaluations/shared-v1-development.json 2>&1 | tee -a evaluate_shared_development.log'"
```

Session: `haetae-shared-dev`; log: `evaluate_shared_development.log`; output: `evaluations/shared-v1-development.json`. Neither evaluator opens a locked test split.

The shared-state evaluation completed with report SHA-256 `8dbe775cd09539b9d366047aa886717a9344dd2a271aa65dd030663b0f8a1d97` and file SHA-256 `960349119103216be9b0945565e578ca1a5303afd70b0338ef2c64b92e4afd23`. Its fitted temperature is `1.5197255188671874`. Common-clean source-macro accuracy is `0.5990` on decision development, `0.4869` on transfer development, and `0.8088` on Korean development. Shared packing reduces unpadded decision-development tokens by `8.59%` and Korean-development tokens by `17.56%`; these remain token-count results rather than latency claims.

The accepted paired comparison has report SHA-256 `19c1d1493ef521e0e339697ee1f6c9e845f6390535f0c94a8e27be6537c6ee43` and file SHA-256 `4d81804932a96afadd568ff4668c44c7cd993bd9ba8587ddc6042e015908b343`. Shared minus baseline source-macro accuracy is:

- decision: `+0.0998`, parent-bootstrap 95% interval `[+0.0686, +0.1322]`;
- transfer: `-0.0228`, interval `[-0.0570, +0.0126]`;
- Korean: `+0.3881`, interval `[+0.3684, +0.4073]`.

Decision and Korean NLL and both Brier differences favor shared with intervals excluding zero. Transfer source-macro NLL is `+0.0789`, interval `[+0.0464, +0.1117]`, and Brier is `+0.0330`, interval approximately `[+0.0123, +0.0534]`, so transfer calibration is materially worse despite the accuracy interval crossing zero. The largest transfer accuracy losses are emotion `-0.2845` and offensive-tweet detection `-0.1375`; the Score transfer cell improves by `+0.2250`. All 89 repository tests pass offline. No locked test was opened.

## Pinned Laya external baseline

The independent Laya audit pins runtime commit `d113dca2512fb3eaca313534bc54c7162d87c1d4` and model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Laya is a real open-weight typed-decision model rather than an API wrapper. It encodes one repeated state-plus-question sequence per question, adds two full-sequence transformer layers, scores runtime-defined option markers, and batches all question rows into one model invocation. It does not implement Haetae-style shared-state execution, and Jev's unpublished architecture prevents a structural-equivalence conclusion.

On the audited public development memberships, English Laya task-source-macro hard-label accuracy is `0.6144` versus Haetae shared `0.4869`; Korean multilingual Laya is `0.6281` versus Haetae shared `0.8088`. Laya wins eight of eleven English task-source cells, while Haetae wins all four Korean cells. Unknown Laya training membership prevents a certified zero-shot claim. The original Laya training mixture and base-training artifacts are also unpublished, and the inspected reward implementation is not globally strictly proper because its log component is clipped. These results do not justify using Laya as a teacher, copying its extra layers, or attributing its behavior to RLCD.

A separately frozen authoritative-logit calibration study completed under protocol self-digest `5937ac1ad6aff9a0f6a4a5f41a36605b198858cc935192136929b3aae3b12e13` and received `LAYA CALIBRATION RESULT QUALIFIED` from ChatGPT 6 Pro. Both calibrators were published before either development artifact; all 7,912 calibration/development rows, 5,764 frozen API replays, point metrics, and four 20,000-draw parent-bootstrap comparisons were independently reproduced. No locked test was opened.

The result is asymmetric. On English development, the new scalar has fitted-minus-shipped task-source-macro NLL `+0.0420`, interval `[-0.0233, +0.1032]`, and Brier `+0.0427`, interval `[+0.0054, +0.0784]`; the English Noul logistic worsens both proper scores and lowers overall Noul accuracy from `0.7750` to `0.6964`. On Korean development, the scalar differences are NLL `-0.3341`, interval `[-0.3718, -0.2968]`, and Brier `-0.1073`, interval `[-0.1184, -0.0962]`. The Korean Noul logistic raises overall Noul accuracy from `0.5813` to `0.7347` and improves macro NLL/Brier by `-0.5975` / `-0.2869`, with entirely negative intervals.

This does not isolate a language effect. Sixty compositional formula tags receive 75% of the English equal-task-source fitting objective, and English calibration and development share no task-source names. Korean calibration and development use the same four task types on disjoint parents. The defensible conclusion is that the declared English cross-task-family recalibration transfers worse than Laya's shipped mapping here, while the declared Korean within-task-family calibration repairs probability scale and a large Noul threshold bias. Representation quality, probability scale, and operating thresholds must remain separate measurements.

The predeclared cached-logit English weighting ablation is also complete and received `WEIGHTING ABLATION RESULT QUALIFIED`. Equal origin, equal task within origin, and equal question within task changes the scalar temperature from `14.6079` to `3.9027`. On the same named development population, hierarchical-minus-shipped task-source-macro NLL is `-0.0617`, interval `[-0.1085, -0.0188]`, and Brier is `-0.0209`, interval `[-0.0416, -0.0008]`. Against the old scalar, the differences are `-0.1037` NLL and `-0.0635` Brier, both with entirely negative intervals. All point metrics and both 20,000-draw parent bootstraps reproduced exactly from cached logits; no new inference or locked-test access occurred.

This establishes sensitivity to task taxonomy and weighting, not a uniform gain. The hierarchical scalar improves NLL and Brier over shipped on seven of eleven task sources but worsens PAWS, QNLI, SciQ, and offensive-tweet detection. Emotion supplies most of the macro-NLL improvement, and the shipped Brier interval ends only slightly below zero. Because the development output was already known before this hypothesis, the scalar remains an exploratory calibration candidate. It does not change Haetae's weights or the active execution decision.

## Active execution benchmark

ChatGPT 6 Pro reviewed exact commit `581203961b42b72565e0293e26f34ccf50660e16` and returned `EXECUTION BENCHMARK HOLD`. It reproduced five pre-measurement defects without opening a locked test: missing schedule binding in memory metadata; last-observation overwrite and workload-level rather than parent-level resampling; incomplete semantic validation of artifacts; unsafe recovery of torn or stale equivalence journals; and arm rotation tied to changing traversal order.

The corrected runner now:

- freezes one request schedule for all repetitions and a separate exact memory schedule;
- selects distinct evaluation parents and records both workload and parent identities;
- averages every repeated workload observation before jointly resampling actual parent clusters;
- binds and validates every row against the protocol, workload, schedule, checkpoint, device, exact observation inventory, positive duration, derived timing values, memory counters, and frozen runtime;
- recomputes memory residual decisions instead of trusting a stored Boolean;
- records process identity that remains stable within one process and requires distinct processes across timing repetitions;
- uses a bound equivalence journal, discards only an unfinished final transaction, rejects corruption in committed data, and rejects retained rows from another protocol;
- rotates arms from a stable condition identity, independently of traversal order;
- binds the fixed temperature to the accepted public-development report.

The review's false-pass example now reports the correct P/B ratio `1.028` instead of `0.800`, and reversing input rows leaves the result unchanged. A result-blind public-development preflight freezes 3,963 requests and 7,227 questions, with 256 distinct multi-question parents, 64 distinct single-question control parents, and identical request schedules across all three repetitions. These are implementation checks, not benchmark results.

A second pre-measurement review confirmed those timing, memory, journal-binding, and counterbalancing corrections, then reproduced one remaining evidence-integrity defect: a resumed equivalence transaction could retain logits that contradicted its recorded probabilities, and summary validation did not require the frozen candidate count. The runner now stores authoritative reference and candidate logits for every comparison, derives every probability, log-probability error, action, margin, and difference from those logits during both recovery and summary, binds base candidates to the recorded packed logits, requires the frozen candidate count, and verifies a digest on every completed request transaction. Regression tests cover the resumed contradictory-logit case, candidate-count substitution, and probability underflow. All 107 repository tests pass offline.

ChatGPT 6 Pro reviewed exact commit `73d2b49c3d95267b135adbcb3af5f53ede3426d3` and returned `EXECUTION BENCHMARK START` after 61 controlled probes. It exercised actual journal recovery, resume, final summary, recomputed-digest corruption, candidate-count substitution, ordinary and subnormal logits, every stress category, and an independent scalar log-sum-exp reference. It reported no remaining material pre-measurement blocker and opened no locked test.

The actual benchmark directory is freshly frozen at `evaluations/shared-execution-v1`. Protocol SHA-256 is `c2da2300f2799c4a919e365f741c8dd21c170182ed0d69d1c3599cd553dca654`, workload SHA-256 is `13acd9555efb792ac74d7ab673407275001cce0c337be15cdf8abc7fa8d00622`, and schedule SHA-256 is `f3151b7233205145fa889c8c8c6aff28fb6e4b01ca6e36ab5662059e9cb54025`. It contains 3,963 public-development requests and 7,227 questions, with 256 multi-question parents, 64 single-question controls, 96 diagnostics, and `locked_test_opened: false`.

The resumable execution script is `evaluations/shared-execution-v1/run.zsh`. It runs MPS then CPU equivalence, three fresh-process timing repetitions per device, P/B/S memory passes per device, and final summary. Completed stages are skipped by their bound metadata files; an interrupted equivalence stage resumes from its validated journal.

Launch and resume command:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-shared-execution-v1 "zsh -lc 'set -o pipefail; /Users/minkyu/workspace/haetae/evaluations/shared-execution-v1/run.zsh 2>&1 | tee -a /Users/minkyu/workspace/haetae/shared_execution_v1.log'"
```

Session: `haetae-shared-execution-v1`; log: `shared_execution_v1.log`; stage artifacts: `evaluations/shared-execution-v1/*.meta.json`; final report: `evaluations/shared-execution-v1/summary.json`.

The MPS equivalence stage completed 3,963 requests and 22,537 bound records. Independent validation accepted all 14 same-device groups. Maximum total variation is `8.97131840676968e-06`, maximum absolute log-probability difference is `4.1134871864301203e-05`, and both material and near-tie action changes are zero. Artifact SHA-256 is `d5a70bdd414cbf16700115708d726f73509c7b8e40c33e0b0b6550bd032c6162`.

The CPU equivalence stage also completed 3,963 requests and 22,537 bound records, with all 14 groups passing. Maximum total variation is `5.123981406701672e-06`, maximum absolute log-probability difference is `1.9270702280138607e-05`, and action changes are zero. Artifact SHA-256 is `df53ef8697a7be696caecbc40100e8c137c48c5d986216532c9206e82c12ad28`. The descriptive CPU-versus-MPS packed comparison has zero action changes across 7,227 questions; its maximum total variation is `8.140450314960464e-06` at temperature 1 and `5.769615788867033e-06` at the fixed fitted temperature.

All six timing artifacts completed in distinct processes and passed schedule, row, duration, derived-value, runtime, and artifact validation. Under the frozen aggregate gate, MPS packed-versus-batched complete-request timing passes for both primary populations: decision mean ratio `0.6093`, parent-bootstrap 95% interval `[0.5872, 0.6317]`, p95 ratio `0.4904`; Korean mean ratio `0.6936`, interval `[0.6681, 0.7203]`, p95 ratio `0.4427`.

The per-process MPS mean ratios vary substantially: decision is `0.8759`, `0.3756`, and `0.8151`; Korean is `1.0322`, `0.4121`, and `1.0534`. The pooled frozen gate passes, but this run-order-sensitive spread requires explicit final review before claiming a stable practical speedup.

All six memory passes completed their frozen 1,000-request schedules and passed semantic validation. MPS P, B, and S each have zero residual live-tensor increase and no resource-regression flag. Their observed driver-allocation maxima are `1,116,422,144`, `1,108,033,536`, and `1,108,033,536` bytes respectively; these are operation-boundary sampled maxima rather than guaranteed transient peaks.

The final summary passed an independent full replay of every artifact binding, file digest, exact inventory, numerical gate, runtime identity, and six-process freshness check. It records `equivalence_passed: true`, `timing_passed: true`, `memory_passed: true`, and the predeclared outcome `equivalent_and_faster`. Report SHA-256 is `e93ae819865df02d46c733456722ba69fc76f10090153c07313d075f0c991ce4`; summary file SHA-256 is `2de3e4fbde1ce93a333eb9c2de006673d333ac7a6cbb6da39210b52dad7724aa`.

The complete 37-file review archive has SHA-256 `289435618aaeaa871668ee126238bd0e3bd17b8e67c3e9ccfe296d3e015af19d`. ChatGPT 6 Pro inspected it in the existing Aside REPL conversation together with exact code commit `73d2b49c3d95267b135adbcb3af5f53ede3426d3` and returned `EXECUTION RESULT QUALIFIED`.

The review independently verified the archive hash and all 37 members, all 35 summary-referenced artifact hashes, the protocol and schedules, 7,926 completed equivalence transactions, 103,200 comparisons recomputed from authoritative logits, 14,256 timing rows, six distinct timing processes, 3,000 memory observations, and the complete summary path. It reproduced the historical predeclared outcome `equivalent_and_faster`. No locked test was opened and no new model inference or production timing was run during review.

The equivalence result is sound, but the batch-one MPS practical-speedup claim is not stable enough. Repetition 1 slowed the complete-request B arm on 94 of 100 decision requests and 99 of 100 Korean requests by more than 1.5 times their repetitions-0/2 average; median inflation was `2.02` and `2.43` times. Excluding that repetition only as an exploratory sensitivity gives decision P/B `0.8461` and Korean P/B `1.0429`. These values do not replace the frozen analysis, but show that one execution condition controls much of the pooled effect.

The Latin rotation correctly gives every arm each nominal position once. It does not equalize complete execution history: repetitions 0 and 2 run pretokenized batch-one work before complete-request timing, while repetition 1 runs complete-request timing first. The estimator correctly averages workloads and resamples parents, but its interval is conditional on the three recorded processes and does not measure process-history variability. Repetition 1 contributes `50.3%` of the pooled B denominator for decision and `55.4%` for Korean.

The controls narrow the interpretation. MPS single-question P/B stays near one without a matching slowdown. CPU multi-question ratios are stable and below one. MPS batch-two ratios are stable and below one for both primary populations. MPS pretokenized decision ratios are stable below one, while Korean is approximately neutral. The strongest supported conclusion is therefore that packed and batched execution are numerically equivalent on the audited population; CPU multi-question and batch-two MPS execution show replicated benefits; batch-one MPS decision is promising; and a stable batch-one Korean advantage is not established. The zero residual memory result remains a residual-liveness check, not a transient-peak comparison.

## Batch-one MPS confirmation

The additive MPS-only batch-one confirmation implementation is commit `98e269f2e242cf940ebf7c2484600cf66a2eb2bf`. All 113 repository tests pass. Its frozen directory is `evaluations/shared-execution-confirmation-v1`:

- protocol SHA-256: `500801125e7a0056cde7d6cafdc838b7463eed9dafa45ffaba54b6e78cede1ea`;
- schedule SHA-256: `1903d67bbfa065bb1155e9df60a15edf41902880e51fed511ccd7fbf92fac911`;
- source checkpoint generation 79 and SHA-256 `e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c`;
- source protocol, workload, schedule, and qualified-summary bytes are bound unchanged;
- each of six processes contains the same 100 decision, 100 Korean, and 64 single-question-control requests;
- every request is P-first in three processes and B-first in three, while each process has exactly 50/50/32 P-first requests in the three populations;
- each fresh process gives both arms two full untimed complete-request passes, then measures one adjacent P/B pair per request before any other benchmark scope;
- the report preserves the original equal-workload and parent-bootstrap gates, reports crossed process-and-parent uncertainty, requires P/B below one in at least five processes and in both arm-order strata, and fails when the single-question-control point ratio leaves `[0.9, 1.1]` or its crossed interval excludes one;
- Historical pre-run state: at freeze time, no checkpoint inference had run under this confirmation protocol, and no locked test had been opened.
- completed process markers are semantically revalidated before they are skipped; an unbound result without metadata is preserved as incomplete and rerun in a fresh process; combined observations can resume only when their bytes reproduce exactly; the final summary has a full replay command.

Pre-run review and measured execution:

The pre-measurement review archive is `/Users/minkyu/workspace/haetae-shared-confirmation-prerun-v3.zip`, with SHA-256 `efae5b18cb1c373b3521133245685f36e12a51ecb7e5bdb157101bd2f3bc042e` and manifest SHA-256 `e8d2913c1b6a6da6ed06fde8c983e3179533058f4870f95d345e3577a6d6e2c9`. ChatGPT 6 Pro verified the archive, commit, frozen bindings, schedule balance, six-process pipeline, synthetic evidence, recovery behavior, and gates without model inference, then returned `BATCH-ONE CONFIRMATION START`.

The approved resumable six-process script was launched in tmux:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-shared-confirmation-v1 "zsh -lc 'set -o pipefail; /Users/minkyu/workspace/haetae/experiments/run_shared_execution_confirmation.zsh 2>&1 | tee -a /Users/minkyu/workspace/haetae/shared_execution_confirmation_v1.log'"
```

The approved run started at 2026-09-21 04:06:42 KST.

- tmux session: `haetae-shared-confirmation-v1`
- log: `/Users/minkyu/workspace/haetae/shared_execution_confirmation_v1.log`
- output: `/Users/minkyu/workspace/haetae/evaluations/shared-execution-confirmation-v1`
- resume command: the exact tmux command above; completed process metadata are revalidated and skipped.
- current status: six-process measurement and exact local replay complete.

All six fresh MPS processes completed with distinct identities and 528 bound observations each. The exact verifier reproduced the complete summary.

- summary self SHA-256: `494d1b55cb8cc6e3e8b954b171e525fc569af5509e94351366b4e186717ed207`;
- summary file SHA-256: `c71787fd3c8063f2425a6a9e2f55d5090705865904454257d1e0b18d9389563b`;
- combined observations SHA-256: `39eb188baf99d54ef6eace74a9181e1a9c5cbffd21faa47813e750885e38df5e`;
- decision P/B: `0.7634`, parent-bootstrap interval `[0.7477, 0.7804]`, crossed process-and-parent interval `[0.7450, 0.7827]`, p95 ratio `0.6865`, and all six process ratios below one;
- Korean P/B: `0.8364`, parent-bootstrap interval `[0.8171, 0.8569]`, crossed interval `[0.8147, 0.8602]`, p95 ratio `0.7911`, and all six process ratios below one;
- decision order strata are `0.7609` B-first and `0.7659` P-first; Korean order strata are `0.8404` and `0.8324`;
- the single-question control P/B is `0.9881`, parent-bootstrap interval `[0.9723, 1.0045]`, crossed interval `[0.9268, 1.0525]`, and p95 ratio `0.9952`.

The frozen gates all pass and the recorded outcome is `batch_one_speedup_confirmed`. One diagnostic requires explicit final interpretation: every single-question control has identical packed and batched model input, so the second member of each adjacent pair immediately repeats the first input. Its pooled counterbalanced ratio is neutral, but its order-stratum P/B ratios are `0.5568` when B is first and `1.7538` when P is first; the first operation averages `1.7748` times the second. This does not appear in either primary population, whose packed and batched shapes differ and whose order-stratum ratios agree closely. Final review must decide whether the exact counterbalance and primary-stratum stability adequately explain this control-order effect.

No locked test was opened. The original execution report remains unchanged.

The measured review archive is `/Users/minkyu/workspace/haetae-shared-confirmation-measured-v1.zip`, with SHA-256 `3001cfdb8af8ab336dc371ef041f7f2b046218639604abdc6709455e450946f7` and manifest SHA-256 `4dea7522bd877ac5f1065d2c9b9edd3fd9f349afcb586edb15d2b0c8e576542c`.

ChatGPT 6 Pro independently verified the measured archive and returned `BATCH-ONE CONFIRMATION RESULT QUALIFIED`. It matched the archive SHA-256, all 39 ZIP members, all 38 manifest-listed payloads, every frozen binding, all 3,168 observations and 1,584 adjacent pairs, the six distinct process identities, every point statistic, all six bootstrap intervals, the frozen gates, and the exact final replay. It ran no new timing or model inference and opened no locked test.

The qualified claim is deliberately narrow: on the frozen public-development multi-question workloads after the declared warmup, packed execution reduced mean complete-request latency by approximately 24% for decision and 16% for Korean on the recorded Apple M3 Pro MPS environment. The result replicated across all six fresh processes and both adjacent-arm orders. It is stronger evidence than the original three-repetition benchmark, whose pooled batch-one result depended heavily on one anomalous process.

The single-question control's position effect is real and remains part of the result. Its first operation averaged 28.088 ms and its immediately repeated second operation averaged 15.826 ms. Exact counterbalancing makes this effect label-neutral in the pooled control, and both primary populations favor packed execution in both order strata. ChatGPT 6 Pro also performed a post-hoc same-position sensitivity check: first-position-only P/B is `0.7661` for decision, `0.8637` for Korean, and `0.9879` for the identical-input control. This sensitivity supports the scoped interpretation but is not a predeclared acceptance gate and does not identify cache, allocator, compilation, or another runtime mechanism as the cause.

The result does not establish cold-start, transport, concurrent-serving, isolated-arm, unseen-shape, cross-machine, cross-session, or universal performance. The Korean aggregate also does not imply the same improvement for every Korean source: exploratory source-level ratios differ substantially. Any deployment-speed claim needs a separately frozen arm-isolated serving trace. No corrective rerun is required for the accepted warmed adjacent-workload claim.

Next research decision:

Freeze a public-development coverage experiment focused on the large emotion and offensive-tweet transfer losses before changing the architecture or starting another training run. The experiment should distinguish task and label coverage from probability calibration and representation error, bind its comparisons before calculating new outputs, and receive ChatGPT 6 Pro pre-run review. Preserve the qualified execution and Laya calibration artifacts unchanged, and do not open either locked test.

## Coverage audit and fixed-weight transfer diagnostic

The derivative-evidence audit is complete through M1. ChatGPT 6 Pro accepted the corrected archive bound to commit `35ee1327b6d53a722d953f067e4df081f88ee6c7`; commit `3c3503d` subsequently stabilized diagnostic NLL for equal logits with very large common offsets. The accepted audit found no emotion or offensive supervision in shared-v1 and no cross-role state or parent overlap.

The first T04 pre-inference review of commit `efdcf8a7f767a665490603b9ba766e1d653b57ff` returned `M3 FIXED-WEIGHT DIAGNOSTIC HOLD`. It reproduced four blockers without loading a checkpoint or running a real model: a 92-question native emotion probe was compared with a 116-question historical baseline; a self-consistent plan could contradict support exclusions or declared fitting settings; semantic macro-F1 used candidate positions rather than a shared label ontology; and the runner split valid JSONL containing a literal Unicode line separator.

Commit `399509c3b02ec55859c37377881dec4595006575` fixes those blockers. The freezer now binds the exact native populations and cached report identities, including keyed membership digests. Verification reconstructs support selection from both pinned Parquet files and reconstructs development membership from the registered public evidence. It requires exact exclusion, split, fitting, compute, model, and evidence contracts. The runner requires the externally reviewed manifest and protocol identities before model lookup, uses the duplicate-key-rejecting byte JSONL reader, and omits semantic macro-F1 while retaining fixed-ontology probe macro-F1. Thirty-three coverage tests and all 146 repository tests pass.

The corrected freeze is `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-m3-v7`. A second independent freeze at `fixed-weight-diagnostic-v1-m3-v8` is byte-identical. The plan manifest SHA-256 is `ea41f5f7a3ba799dc14c24ce3368c38c97684d4c4bf9bafff993dcc6e8a1a663`; the protocol SHA-256 is `e376a60ebbe52d7a8c0129a78c50d2ca4b3950a225089146c4d1ed6dc83b7201`. It contains 2,048 support-fit rows, 512 support-calibration rows, 196 development questions, 1,100 semantic variants, 6,540 encoded sequences, and 3,270 planned microbatch-two forward calls. At freeze and pre-run review time, no model forward, optimizer update, or locked-test access had occurred under this protocol.

The private review archive is `/Users/minkyu/workspace/haetae-artifacts/haetae-fixed-weight-diagnostic-m3-v3.zip`, SHA-256 `36174fd0888a948d67e22a79cf3e938ec096154e69382621364a6cb9af1e5924`, size 5,507,492 bytes. Its 14 members pass CRC and per-member SHA-256 validation. It includes the frozen plan, the two pinned support Parquet files, the registered transfer development input, both cached development reports, and the evidence registry so the reviewer can reproduce exact membership without model inference. It is a private review artifact and is not tracked by Git.

ChatGPT 6 Pro returned `M3 FIXED-WEIGHT DIAGNOSTIC START` for the exact commit, archive, manifest, and protocol identities above. It independently verified all 14 ZIP members and 13 indexed payloads, reconstructed both support selections and all 101 removals, reproduced the 92/12/12/80 development strata and historical values, regenerated all 1,100 semantic variants, and passed 47 independent regression and adversarial checks. It loaded no real checkpoint, ran no real model forward, and opened no locked test. The approval permits one MPS process for this frozen diagnostic only.

The execution uses a detached exact-review worktree so the state-only follow-up commit cannot change the reviewed code identity:

```bash
cd /Users/minkyu/workspace/haetae-diagnostic-m3-run
test "$(git rev-parse HEAD)" = "399509c3b02ec55859c37377881dec4595006575" || exit 1
FROZEN_PLAN=/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-m3-v7
EVIDENCE_PATHS=/Users/minkyu/workspace/haetae-artifacts/paths.local.json
LOCAL_PATHS=/Users/minkyu/workspace/haetae-artifacts/diagnostic-v1-paths.local.json
NEW_OUTPUT=/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-results-v1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src uv run --frozen python -u -m experiments.coverage_v1.run_diagnostic_v1 --plan "$FROZEN_PLAN" --paths "$EVIDENCE_PATHS" --local-paths "$LOCAL_PATHS" --registry research/evidence/index.json --out "$NEW_OUTPUT" --device mps --reviewed-manifest-sha256 ea41f5f7a3ba799dc14c24ce3368c38c97684d4c4bf9bafff993dcc6e8a1a663 --reviewed-protocol-sha256 e376a60ebbe52d7a8c0129a78c50d2ca4b3950a225089146c4d1ed6dc83b7201 --allow-reviewed-inference
```

- tmux session: `haetae-fixed-weight-diagnostic-v1`
- log: `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-results-v1.log`
- output: `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-results-v1`
- resume policy: the diagnostic publishes only a complete new output directory and has no partial checkpoint; if interrupted before publication, preserve the log, diagnose the exact failure, and rerun the same frozen command only after confirming no writer and no output directory exist.

The approved process completed and published all five expected output files. Output manifest SHA-256 is `ed76876fd9ed4aa6ff9824a1c73186ebb9f2113a5cc282cc4598c3abdfd62d23`; result SHA-256 is `861b63a15427e56719e23d9917712de819cad0b721a0685f3c31d44476e0bc23`. It records 6,540 encoded sequences, 3,270 model forward calls, zero optimizer updates, and no locked-test access. An independent replay verified every file digest, safetensors metadata and finiteness, feature-row and semantic-variant alignment, all probe and semantic metrics, both 20,000-draw parent bootstraps, every decision flag, and both output self-digests.

The fixed-feature probes do not show an accessible-feature regression. Calibrated accuracy is `0.5435` original versus `0.5217` shared for emotion, with original-minus-shared interval `[-0.0968, 0.1413]`; offensive is `0.5000` versus `0.5625`, with interval `[-0.1750, 0.0500]`. The shared probes recover slightly more than half of the matched historical accuracy gaps, and support calibration improves probability scores without changing ranking.

The semantic portion has a material unresolved anomaly. All 1,100 stored semantic logit vectors are exactly zero. The 196 original-family rows match every request and question key in the registered cached shared development report, while every matching cached vector has nonzero span, reaching `8.1426`. The checkpoint pointer-head tensors are nonzero. The resulting semantic accuracy, rotation changes, and zero total variation are therefore tie-driven and must not be treated as model behavior until the mismatch is explained.

The private measured review archive is `/Users/minkyu/workspace/haetae-artifacts/haetae-fixed-weight-diagnostic-m3-measured-v1.zip`, SHA-256 `4199afc3538898a0b1b71a6f801557b24ab3b04d584794970cc926b8f0bd71fb`, size 9,657,020 bytes. Its 13 members pass CRC and indexed SHA-256 validation.

ChatGPT 6 Pro returned `M3 FIXED-WEIGHT DIAGNOSTIC RESULT HOLD`. It independently verified the archive and all 12 indexed payloads, output bindings, both feature tensors, all rows and variants, probe metrics, both bootstraps, all semantic groups, and decision flags. The saved-feature probe analysis remains usable. Every semantic conclusion is withheld.

The review proved that the original-family inputs match the 196 historical requests on state, instruction, ordered descriptions, keys, labels, primitive, and parent; each historical request contains one question. Historical within-vector spans range from `0.0261` to `8.1426`, while every one of the 6,068 saved semantic scalar logits is exactly zero. Uniform probabilities and first-index tie breaking reproduce every semantic field in the held result. This is an unresolved runtime or output-evidence failure, not a valid rendering difference or evidence of architectural failure.

The first suspect is the runner's fused `MPS -> CPU float64` conversion. The historical evaluator first copies to CPU and then converts to float64. The review did not claim that this visible code-path difference proves the native failure. It authorized one tensor-only MPS transfer regression with 18 synthetic cases: vector lengths 2, 6, and 7; storage offsets 0, 1, and 256; and `no_grad` versus `inference_mode`. The probe must compare fused and staged transfers with the CPU-created reference, verify source preservation, record exceptions as failures, access no checkpoint or dataset, and run zero model forwards. Do not rerun any real request or the full diagnostic until this tensor-only result is reviewed.

The first implementation at commit `13de58381911dcbf7066914c2f0dafe0575712ee` received `MPS COPY DIAGNOSTIC HOLD`. Its 18-case boundary and finite failure handling passed, but a returned NaN or infinity reached strict JSON serialization and prevented publication of the entire failure report. The correction preserves strict JSON, compares tensors before encoding, serializes nonfinite observations as `NaN`, `+Infinity`, or `-Infinity` strings, and adds an end-to-end regression proving that one nonfinite case plus 17 valid cases publishes a complete self-digested failed result and exits nonzero at the command boundary.

Commit `e50e0b209782368f7028372abb44c9fc50290aee` closes the nonfinite-reporting defect. ChatGPT 6 Pro reviewed the exact commit, ran four supplied tests and 38 independent controlled checks without native MPS or model access, and returned `MPS COPY DIAGNOSTIC START` for one bounded execution.

The authorized tensor-only diagnostic ran once and was preserved at `/Users/minkyu/workspace/haetae-artifacts/mps-copy-diagnostic-v1.json`. Its file SHA-256 is `caa7028ef112402c252f7896d3053c83ff64b92fa6e883cd42bcf8324e344b8a` and its self-digest is `5fac32c20a2a5c737786bfecf2929b3cbb77df1261011372c64bcb8d90f56c3d`. It exited with code 1 because all 18 cases reproduced the same defect:

- the fused MPS-to-CPU float64 conversion returned an all-zero vector without raising an exception;
- copying to CPU first and then converting to float64 exactly matched the CPU-created reference;
- the source tensor remained unchanged and its storage offset matched in every case;
- the behavior was identical for lengths 2, 6, and 7; offsets 0, 1, and 256; and both `no_grad` and `inference_mode`;
- no checkpoint, dataset, model forward, optimizer update, or locked test was accessed.

This isolates the held semantic result to its output conversion path. The saved-feature probe result remains usable, while the old semantic result remains preserved and held. An output-adapter-only amendment was submitted to ChatGPT 6 Pro to retain the validated feature artifacts, rerun only the frozen 1,100 semantic variants, require exact membership plus predeclared numerical parity for all 196 original-family rows against the registered generation-79 development report, and publish a new bound result. No further real-model execution is authorized until that implementation receives exact-commit pre-run review. The separately frozen 32-request exported-runtime parity check remains after T04 is resolved.

ChatGPT 6 Pro independently verified the tensor result and returned `OUTPUT ADAPTER AMENDMENT ACCEPT`. The amendment keeps the historical producer and held output unchanged, loads only generation 79, performs no feature extraction, and uses `logits.detach().cpu()` before widening the CPU float32 values to float64. It stores both representations for every semantic observation and rejects nonfinite, wrong-shape, wrong-dtype, or inexactly widened evidence.

The mandatory original-family gate covers all 196 matched requests. Every raw logit must satisfy `abs(candidate - reference) <= 1e-4 + 1e-5 * abs(reference)`. At temperatures 1 and `1.5197255188671874`, maximum absolute log-probability error must be at most `1e-3`, maximum total variation at most `1e-4`, and nearest-rank P99 total variation at most `1e-5`. No action may change when the cached reference top-two probability margin exceeds `2e-4`; near-tie changes remain explicitly reported. These tolerances are frozen before corrected inference and cannot be relaxed after observing a result.

The versioned implementation is `experiments/coverage_v1/semantic_output_amendment_v1.py`, with CPU-only checks in `tests/coverage_v1/test_semantic_output_amendment_v1.py`. All 160 repository tests pass. The amendment preflight validates the original plan, held output, saved float32 features, cached public-development report, tensor diagnostic, code, checkpoint, tokenizer, runtime, and exact 1,100-row schedule without loading a checkpoint or running a model.

The frozen amendment is `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-m3-v1`:

- amendment manifest SHA-256: `f106ac3d6841e6339f4e9135b144167dafa453c4856b1dcde518e29475e72059`;
- amendment protocol SHA-256: `4b5358296b5828e87da6c9308f0e33a1e32eed059d02501f7b2021be1f86621b`;
- planned new execution: one MPS process, 1,100 semantic sequences, microbatch two, at most 550 forward calls, zero feature forwards, zero original-pretrained-model loads, and zero optimizer updates;
- no real checkpoint was deserialized and no MPS model forward or locked-test access occurred while implementing, testing, or freezing the amendment.

Next action: commit and push the exact amendment implementation, submit that commit plus the frozen manifest and protocol identities to ChatGPT 6 Pro for pre-run review, and execute the single semantic-only MPS process only if it returns an explicit start decision.

The first exact-commit pre-run request covered commit `7ee17c3e4d4ba758055e7f457c9e434fd080bb77` and amendment freeze `fixed-weight-diagnostic-semantic-amendment-v1-m3-v1`. ChatGPT 6 Pro verified the archive hash, size, all 20 members, CRC, index, GitHub source, and controlled failure paths, but its response stopped before returning a terminal start or hold decision. This is not execution authorization, and no real-model work followed it.

A local fail-closed audit then tightened the amendment before resubmission:

- serialized semantic evidence must contain exactly four fields, have `status: complete`, and preserve the frozen variant order;
- the generation-79 loader now also requires the exact run ID, specification digest, eager-attention setting, 2,048-token limit, and float32 parameter dtype;
- the CPU verifier reconstructs the complete reuse and scientific result objects rather than checking selected fields;
- execution receipts have an exact schema and fixed new, reused, and cumulative counters;
- a failed final replay deletes any complete manifest before publishing the failed receipt, so a failed verification cannot leave a complete-looking result;
- amendment, plan, held-output, and result directories reject symlink substitution;
- nested nonfinite or malformed tensor evidence is preserved in strict JSON.

All 160 repository tests and static checks still pass. The corrected freeze is `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-m3-v2`:

- amendment manifest SHA-256: `d8cfa217a5be39769550f11a8984fb947bd8a175f1a1ba74a056ac36e0c6ad0a`;
- amendment protocol SHA-256: `d7992de4cc9c43ebbefe700ed6860d6d808541c98a16b67f7bad30f7ce2e5976`;
- no checkpoint deserialization, model forward, optimizer update, or locked-test access occurred.

ChatGPT 6 Pro returned `SEMANTIC OUTPUT AMENDMENT HOLD` for commit `d744eeccccdfcd24062ceccb50012703165530e2`. It verified the replacement archive and reproduced two remaining boundary defects without MPS, checkpoint, or locked-test access. Numerical vector validation occurred after the original-family filter, so 904 non-original variants could contain contradictory or malformed stored transfer evidence while the final verifier still accepted recomputed semantic results. The amendment freeze manifest and protocol and the completed-output manifest also bypassed the existing leaf-file symbolic-link check.

Commit `75cbaac35d46a636d86cd4ba9ced9b1f4dea95da` corrects only those two defects. Every one of the 1,100 observations is now checked for the frozen candidate count, finite non-Boolean numbers, exact float32 representability, and exact equality between widened float32 values and stored float64 logits before original-family filtering. Historical-reference tolerances remain restricted to the 196 originals and are unchanged. The three boundary JSON files now pass through the existing explicit allowlist and leaf-symbolic-link check before reading or hashing. End-to-end final-verifier regressions cover contradictory and nested malformed non-original evidence; focused tests also cover non-exact float32 values and all three leaf-file substitutions. All 168 repository tests and static checks pass.

The new freeze is `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-m3-v3`:

- amendment manifest self-digest: `27247ecd3cf2aecf5a9030cd85fd30619941adafd42421436f1320517efd7b77`;
- amendment protocol self-digest: `f23a19e03a226d8c0ceb8c9fbdd1dc9a588ca63050c2f7e04116381157cae92c`;
- fresh freeze replay: 1,100 variants, 2,720 feature rows, and 196 historical references;
- no checkpoint deserialization, model forward, optimizer update, or locked-test access occurred.

The private review archive is `/Users/minkyu/workspace/haetae-artifacts/haetae-semantic-output-amendment-v1-prerun-v3.zip`, SHA-256 `e44b061f72609873e2d4fef2078299532f98cbbb8567b0a54f3a52af787b0e00`, size 10,662,136 bytes. Its 20 members pass CRC validation, and all 19 review-index payload descriptors reproduce. ChatGPT 6 Pro verified the exact archive and commit, reran the previous non-original evidence exploits and the three leaf-file substitutions through CPU-only publication and replay paths, and returned the exact terminal decision `SEMANTIC OUTPUT AMENDMENT START`.

The authorized execution uses the detached exact-review worktree `/Users/minkyu/workspace/haetae-semantic-amendment-run` at commit `75cbaac35d46a636d86cd4ba9ced9b1f4dea95da`. It permits one generation-79 MPS process over only the frozen 1,100 semantic variants, microbatch two, and at most 550 forward calls. It permits no feature extraction, original-pretrained-model load, optimizer update, retry, or resume. The frozen command is:

```bash
cd /Users/minkyu/workspace/haetae-semantic-amendment-run
test "$(git rev-parse HEAD)" = "75cbaac35d46a636d86cd4ba9ced9b1f4dea95da" || exit 1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src uv run --frozen python -u -m experiments.coverage_v1.semantic_output_amendment_v1 run --amendment /Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-m3-v3 --plan /Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-m3-v7 --held-output /Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-v1-results-v1 --copy-result /Users/minkyu/workspace/haetae-artifacts/mps-copy-diagnostic-v1.json --paths /Users/minkyu/workspace/haetae-artifacts/paths.local.json --local-paths /Users/minkyu/workspace/haetae-artifacts/diagnostic-v1-paths.local.json --registry research/evidence/index.json --out /Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-results-v1 --device mps --reviewed-manifest-sha256 27247ecd3cf2aecf5a9030cd85fd30619941adafd42421436f1320517efd7b77 --reviewed-protocol-sha256 f23a19e03a226d8c0ceb8c9fbdd1dc9a588ca63050c2f7e04116381157cae92c --allow-reviewed-inference
```

Execution state:

- tmux session: `haetae-semantic-amendment-v1`;
- log: `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-results-v1.log`;
- output: `/Users/minkyu/workspace/haetae-artifacts/fixed-weight-diagnostic-semantic-amendment-v1-results-v1`;
- failure policy: preserve every partial observation and the failed receipt, do not retry, and return the exact evidence to ChatGPT 6 Pro.

The authorized process completed once and published all six required files. Output manifest self-digest is `1ff4d7e4719e435d67c42d3c3887491ad776df1978fff9d60dc4666bf58494da`; result self-digest is `54322891b1d3781bc8d34c6afba2af037ea853b6a59219adffcf12eba74a58f7`. The receipt records 1,100 complete semantic sequences, exactly 550 attempted and completed model forward calls, one accelerator process, zero feature-extraction forwards, zero original-pretrained-model loads, zero optimizer updates, and no locked-test access. All 1,100 rows and 6,068 scalar logits are finite, exact float32-to-float64 widenings, and nonconstant.

The 196-original parity gate passed. Maximum absolute log-probability error is `9.02970248795043e-06`, with no material or near-tie action changes. At temperature one, maximum total variation is `1.9182769782785875e-06` and nearest-rank P99 is `1.5876989444031653e-06`. At the frozen shared temperature, maximum total variation is `1.2194452650819698e-06` and P99 is `1.0628863641101716e-06`. The exact-commit CPU-only verifier independently recomputed every binding, metric, decision, receipt, and digest and exited zero without model execution.

The optional outer launcher's exit-marker file was not written because its post-command shell assignment used zsh's read-only `status` parameter. This occurred after the scientific command emitted its final complete manifest and does not alter the recorded run. No retry or second model execution occurred. The complete receipt, self-digested manifest, final log record, and independent verifier result are preserved.

The private measured review archive is `/Users/minkyu/workspace/haetae-artifacts/haetae-semantic-output-amendment-v1-measured-v1.zip`, SHA-256 `b78510665d78292c3f8ff2bec1650380831ce37f4373ef03a4fa4357456cbc31`, size 10,813,642 bytes. Its 28 members pass CRC validation, and all 27 indexed payload descriptors reproduce.

ChatGPT 6 Pro returned `SEMANTIC OUTPUT AMENDMENT RESULT ACCEPT`. It verified archive continuity with all 19 approved pre-run payloads, every file and self-digest, both saved feature tensors, all 1,100 observations, all 27 semantic groups, eight probe metric blocks, both 20,000-draw parent bootstraps, and every decision flag. Fourteen altered-evidence cases failed as required. Its different CPU numerical stack produced sub-ULP aggregate differences: parity fields differed by at most `4.44e-16`, semantic metrics by at most `8.88e-16`, probe scores by at most `2.37e-08`, and fitted calibrator parameters by at most `2.86e-07`. All frozen parity and probe replay tolerances passed, and exact discrete metrics, bootstrap endpoints, and decisions agreed. The archived same-environment verifier remains the exact-replay evidence; cross-platform refitting is numerically reproducible rather than byte-identical.

T04 is closed. The corrected semantic evidence shows substantial sensitivity to explicit task descriptions with unchanged weights. On native emotion, the fixed native-taxonomy lexicon improves raw pointer accuracy from `33/92` to `53/92`, reduces NLL from `1.602480` to `1.328946`, and recovers 20 of the 29 correct answers missing from the historical baseline. On offensive questions, it improves accuracy from `39/80` to `57/80`, NLL from `1.070582` to `0.592636`, Brier score from `0.737178` to `0.403866`, and AUROC from `0.648860` to `0.703704`.

The lexicon is not a deployment-ready repair. Offensive predictions move from 63 to 15 positives: true positives fall from 24 to 9, false positives from 39 to 6, false negatives rise from 2 to 17, and recall falls from `92.31%` to `34.62%`. The 71.25% accuracy is only three correct answers above the always-negative rate. Candidate-order and wording sensitivity also remain: 132 of 460 native emotion question-rotation pairs change action, 43 of 92 questions change on at least one rotation, and 24 of 92 change under the alias prefix. The fixed lexicon helps when `none_of_these` is an added candidate but fails the small stratum where it is the correct target. These are measured semantic sensitivities, not the previous all-zero artifact.

Next research decision: prepare the T05 matched targeted-replay pre-training freeze. Do not start an architecture or loss sweep and do not search more prompts on this development population. Both arms start from generation 79, use the unchanged architecture and loss, reset the optimizer, run 512 updates, use effective batch eight, and start with seed 17. The continuation control draws eight original-training requests per update. The targeted arm draws six matched original requests plus one emotion and one offensive support-fit request per update. Freeze both draw tapes, exact task renderings, optimizer reset, checkpoint endpoints, and all analysis before updating weights. Use no development fitting.

The T05 primary comparison uses the original native request wording. The already-frozen lexicon, rotations, and alias conditions are secondary robustness diagnostics with the zero-update generation-79 result retained. Report offensive confusion counts, recall, precision, and AUROC with accuracy. The pilot requires at least five percentage points of targeted-minus-control accuracy improvement on both targets while satisfying the predeclared decision, Korean, and other-transfer nonregression limits. Only a passing seed-17 pilot permits the second-seed replication. This remains exploratory public-development research and requires a new exact-commit ChatGPT 6 Pro pre-run review before training.

## T05 paired replay pre-M4 freeze

The paired-replay implementation is frozen at producer commit `e06ca7362a06a0ea454df06523ac484657db317b` on branch `next/science-coverage`. It adds strict input and file-boundary contracts, deterministic paired tapes, weight-only initialization, current-only checkpoint recovery, terminal endpoint evaluation, parent bootstrap analysis, guarded command dispatch, and a private review package.

The implementation specification self-digest is `a414f002e2b742c94c34f08ef22de4d04ee6cf8ad2b28e237371481272fe2354`. Freeze directories `/Users/minkyu/workspace/haetae-artifacts/t05-paired-replay-v1-freeze-a` and `/Users/minkyu/workspace/haetae-artifacts/t05-paired-replay-v1-freeze-b` are byte-identical. Protocol self-digest is `14b31e11ad408ec6108a8ef49cc0fcd01f8ea586b3459f8354054866562a3c39`; manifest self-digest is `c1b45d42800637b15627cab91fad18b10c110be35ea0ff86e0eb0c9cc12844c0`.

The freeze binds 15,572 original requests with 23,068 questions, 2,048 native target-support requests, four 512-update tapes, and an endpoint inventory of 8,131 vectors from 4,867 requests. The primary population has 7,227 questions; 904 non-original T04 variants remain secondary. All pre-M4 activity counters are zero. Two independent freezes and a third full reconstruction reproduce every scientific byte. All 51 focused tests, Ruff, compilation, immutable-source diff, and the forbidden-term audit pass.

The private M4 archive is `/Users/minkyu/workspace/haetae-artifacts/haetae-t05-paired-replay-v1-prerun.zip`, SHA-256 `44d5dd5dae94d9d6cc9928e7fdb065424810a7fa07ca17c82efe91e744d9f1b7`, size 40,471,214 bytes. It contains 88 members and 87 indexed payloads; review-index self-digest is `06a488ee9d93c55bcc64ba96e6cde05ed141702b532867baac0a60303f8d1b0b`. It contains no checkpoint tensor payload.

No initialization, model forward, MPS operation, checkpoint deserialization, optimizer update, or locked-test access has occurred. Submit the exact producer commit and private archive to ChatGPT 6 Pro through Aside REPL. Only `M4 PAIRED REPLAY START` authorizes the exact sequential seed-17 pair. The current volume has about 33 GB free, below the frozen 32-checkpoint reserve of about 54 GB; execution also requires a non-destructive space recovery before initialization.

## T05 pause boundary

The exact M4 request was submitted in the recorded Aside conversation with archive `haetae-t05-paired-replay-v1-prerun.zip` and corrected producer commit `e06ca7362a06a0ea454df06523ac484657db317b`. At the 2026-09-21 pause boundary, ChatGPT 6 Pro had verified the archive identity and was still running CPU-only code, tape, initialization, checkpoint, evaluation, bootstrap, and seed-23-gate checks. It had not returned either terminal decision. The earlier mistyped commit identity is invalid and must not be used.

Research is intentionally paused at this review boundary. Do not initialize a model, deserialize the checkpoint, run MPS work, update weights, clean scientific artifacts, or start either seed-17 arm while the status is `waiting`. On an explicit resume, first read the completed review response in full. If it returns a hold, reproduce each blocker locally and change only the reviewed defects. If it returns a start, record the decision, recover enough space from regenerable caches or other non-scientific material to satisfy the frozen reserve, create an exact detached execution worktree at the producer commit, and then follow the reviewed sequential seed-17 procedure.

ChatGPT 6 Pro returned `M4 PAIRED REPLAY HOLD` after verifying the corrected producer, complete archive, all four tapes, scheduler table, cached references, and 51 supplied tests. It ran 14 additional CPU-only probes without checkpoint deserialization, model inference, accelerator work, optimizer updates, or locked-test access. The review reproduced seven execution blockers:

1. The pilot gate reads flat target counts while the real evaluator emits `raw` and `fixed` sections, causing analysis to fail after bootstrap work. NLL also underflows on large finite margins, and the bootstrap sampler and contrast inventory differ from the frozen protocol.
2. Seed 23 accepts a re-digested Boolean gate without recomputing endpoint evidence. Its evaluation command also lacks the required pilot-gate argument.
3. Report ingestion validates only a small outer digest boundary and can accept a foreign seed, arm, runtime, checkpoint, temperature, inventory, or malformed observations.
4. Initialization records the source identity before device transfer but does not prove that the transferred and published step-zero tensors remain identical. Training also accepts an under-bound pair receipt or a receipt naming another run.
5. Initialization, training, and evaluation reconstruct the current runtime without comparing it with the reviewed frozen runtime before any model or checkpoint operation.
6. Catchable failures can leave only running events, and a later `running` checkpoint can resume without proof of a graceful interruption or a narrow recovery review.
7. Output-root containment uses lexical absolute paths and accepts parent traversal outside the configured root.

The separate storage blocker remains: the frozen reserve is 53,995,144,544 bytes and the measured free space was about 33 GB. Preserve generation 79, T04 evidence, support membership, the original freeze, and the review archive. Do not start either arm. Repository foundation and release-candidate cleanup now take priority; T05 resumes only after the seven defects are corrected, a new code-bound freeze is produced twice, and the new exact producer receives another M4 review.
