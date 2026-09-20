# Haetae research state

Status: active

## Objective and completion criteria

Build and evaluate a clean-room local System One decision model that is structurally closer to Jev than an NLI wrapper: one packed non-generative forward pass per typed question; dynamic Choice, Noul, and Score outputs; calibrated probabilities; Korean support; and local serving through a Jev-compatible API.

Completion requires a reproducible completed checkpoint, named held-out evaluations, temperature calibration and selective-risk evidence, permutation and option-perturbation stress tests, local CPU and MPS latency, and a ChatGPT 6 Pro review of the final code and measured results.

## Repository and external review

- Repository: https://github.com/pumpkinredbean/haetae
- Branch: `main`
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

The corrected public-only rendered-state audit preserved the existing frozen suite bytes and again found zero overlap across 20,084 model-visible states and 18,412 source-parent identities. Its artifact SHA-256 is `b1d4c966d6a29493a3580f2f8e8b75ddf446f660f6e37e4a792e2b33ca487dbc`, and its file SHA-256 is `fa1300b3ec30ad321bacaa21ec41fcccb658b8b93378a75aef5d22aa739280b3`. All 79 repository tests pass.

A real pinned-mmBERT MPS probe completed one effective batch of eight as four microbatches of two. It deliberately paired the six longest 994–1,011-token, 77-option requests and the two longest multi-question requests. The optimizer update took `2.61` seconds; maximum observed driver allocation was `4.38 GB`, `14.52%` of the `30.15 GB` recommended maximum. A real entry-point smoke run then published an interrupt at step 4, restored generation 6 in a fresh process, continued through step 8 with the exact cursor and random state, and published interrupted generation 11. Smoke run ID is `91b2b05b-d116-41b6-a40f-412c76d36020`; generation 11 SHA-256 is `73c14456db412f3e2683b50a00b31b30fa05450eeb6ff73d4c6ef811db4d78d7`.

The fresh shared-state run is scheduled from 2026-09-20 18:18 KST. It performs exactly two request epochs: `ceil(15,572 / 8) * 2 = 3,894` optimizer updates.

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-shared-v1 "zsh -lc 'set -o pipefail; PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 HF_HUB_OFFLINE=1 uv run python -u -m experiments.train_shared --suite evaluations/shared-v1 --steps 3894 --batch 8 --microbatch 2 --max-len 2048 --device mps --local-files-only --out runs/shared-v1 --save-every 50 --log-every 25 --resume none 2>&1 | tee -a train_shared_v1.log'"
```

- tmux session: `haetae-shared-v1`
- log: `/Users/minkyu/workspace/haetae/train_shared_v1.log`
- run ID: `71d06d48-a3cf-4ddc-bfe2-2315dbe66da8`
- run-spec SHA-256: `2797fc5a985c315f035658790c6ba582d634fe2ceeaa2d8fc5d35e78a100e9d5`
- training-code SHA-256: `ff54ea26399fd9ad335eabba9d6f6a85a49880d8ac23b5217de665b10c2cc3fe`
- generation 1 is the validated running checkpoint at step 0;
- generation 2 is the first durable trained checkpoint at step 50, SHA-256 `2a53635a19f109fdd2ed128aaac373a6186255266f7920eef52f013b71fef40b`; the run continued past step 75 at approximately `1.51 s/update` including checkpoint time.

## Next actions

1. Preserve the validated role C result and use it as the baseline measurement.
2. Obtain a narrow follow-up check of the external-suite binding fix at `aafd461`.
3. Complete the fresh shared-state MPS run from the frozen recipe.
4. Compare the completed baseline and shared-state model on the frozen Kev and Korean development suites without opening either locked test during model selection.
