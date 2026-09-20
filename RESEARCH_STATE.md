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

## Frozen measurement protocol

`haetae.measure` implements the three-role protocol requested by the external review.

- Each source has an explicit labeled evaluation-track registry. Hugging Face sources load at a declared commit revision; local Korean files are bound by SHA-256.
- Preparation reconstructs and verifies the run's consumed training and internal-validation fingerprints. It excludes complete parents with any exact normalized-record or state overlap.
- A, B, and C assignments are deterministic, source-order independent, and disjoint by record, state, and parent. A fits one source-balanced soft-target temperature. B freezes conformal thresholds and a Bonferroni-corrected selective-risk family. Only the explicitly gated final command reads C.
- The final artifact reports stable soft-target NLL, histogram and annotation Brier scores, full-class macro F1, calibration bins, Noul yes-probability reliability, ordinal RPS and expected-index error, parent-cluster bootstrap intervals, conformal coverage and set sizes, and simultaneous selective-risk bounds. Zero accepted examples are recorded as not certified.
- Choice stress uses one formal question per parent and cell across all eligible sources. Permutation and controlled option removal reuse the exact retained state and instruction token prefix. Removal never deletes the gold option or the original top prediction, and binary questions are reported as ineligible rather than divided by zero.
- Latency uses a digest-selected workload and records model-only and packing-inclusive distributions for batch one and the evaluation batch. MPS trials synchronize before and after timing and record workload shape, warmup, repetitions, software, hardware, and accelerator memory.

The protocol has 12 focused tests; all 39 repository tests pass. It also passed two real completed-checkpoint paths. A HelpSteer2 run exercised A/B/C, five response-attribute cells, forced full conformal sets, selective abstention, and synchronized MPS latency. An AG News run exercised 30 semantic option permutations and 10 controlled option removals over 10 C-role parents with no context-prefix drift. These are mechanics smokes, not quality results.

An offline preflight loaded every declared remote source at its exact cached commit plus both local Korean files, reconstructed the old full run's 25,500 training and 3,400 internal-validation questions, and froze every registered track without a role leak. With a 40-parent cap per track, plan `3e48f2d0294a64e9558cbc5bfc07521c875989cbac7a020c5d0a175a3239ec1a` contains A/B/C record counts 274/274/543 and parent counts 140/140/280.

## Next actions

1. Start the reviewed fresh replacement run and monitor its authoritative manifest, process, memory, and periodic generations.
2. Complete the frozen certification protocol with Choice stress tests, synchronized latency, and digest-bound artifacts.
3. After completion, evaluate the baseline on its held-out sources and the frozen `kev` transfer development suite.
4. Design the shared-state architecture as a separate run rather than changing this baseline in place.
5. Send the exact final code commit and measured results to ChatGPT 6 Pro through the same Aside REPL conversation, implement supported findings, and rerun affected evidence.
