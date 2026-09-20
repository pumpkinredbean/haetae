# Haetae research state

Status: active

## Objective and completion criteria

Build and evaluate a clean-room local System One decision model that is structurally closer to Jev than an NLI wrapper: one packed non-generative forward pass per typed question; dynamic Choice, Noul, and Score outputs; calibrated probabilities; Korean support; and local serving through a Jev-compatible API.

Completion requires a reproducible completed checkpoint, named held-out evaluations, temperature calibration and selective-risk evidence, permutation and option-perturbation stress tests, local CPU and MPS latency, and a ChatGPT 6 Pro review of the final code and measured results.

## Repository and external review

- Repository: https://github.com/pumpkinredbean/haetae
- Branch: `main`
- Checkpoint format 3 implementation commit: `a30ba8b`
- Aside conversation: `Clean Room Jev Reproduction`
- Aside URL: https://chatgpt.com/c/6aaeba00-b670-83e8-9c29-3370b3c7945d
- Review model: ChatGPT 6 Pro
- Interaction method: Aside REPL only
- Exact review of commit `a30ba8b`: publication, rotation, AdamW checks, real signal continuation, and epoch rollover passed. Reported semantic-validation, tokenizer-identity, calibration-binding, descriptor-schema, completed-target, recovery-display, and certification-policy findings are fixed in the current working tree.

## Confirmed model design

- ModernBERT-base backbone.
- One row per question: `[CLS] state [SEP] instructions [SEP] option1 [SEP] option2 ...`.
- A shared scalar head reads the bidirectional hidden state of the separator preceding each option.
- Candidate logits are normalized only within their question.
- Choice order is augmented during training. Score order and Noul yes-first semantics remain fixed.
- Hard or soft cross-entropy plus Brier is the baseline objective. Cross-entropy plus ranked probability score for Score is the first planned loss ablation.
- The mixture contains 13 sources, including KLUE-YNAT and NSMC for Korean.
- Calibration claims must identify the dataset and deployment conditions. Selective action uses a one-sided binomial risk bound.

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

## Planned long run

No long training process is active while checkpoint format 3 is under review. After the reviewed code is pushed, start a fresh run:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3 "zsh -lc 'set -o pipefail; HF_HUB_OFFLINE=1 uv run python -u -m haetae.train --sources ag_news,banking77,massive,mnli,anli,arc,emotion,klue_ynat,boolq,nsmc,civil_toxicity,sst5,helpsteer2 --per-source 1500 --eval-per-source 200 --steps 3000 --batch 8 --max-len 1536 --out runs/v3 --save-every 50 --resume none 2>&1 | tee -a train_v3.log'"
```

Resume the same run only with:

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae-v3 "zsh -lc 'set -o pipefail; HF_HUB_OFFLINE=1 uv run python -u -m haetae.train --sources ag_news,banking77,massive,mnli,anli,arc,emotion,klue_ynat,boolq,nsmc,civil_toxicity,sst5,helpsteer2 --per-source 1500 --eval-per-source 200 --steps 3000 --batch 8 --max-len 1536 --out runs/v3 --save-every 50 --resume auto 2>&1 | tee -a train_v3.log'"
```

- tmux session: `haetae-v3`
- log: `/Users/minkyu/workspace/haetae/train_v3.log`
- authoritative run identity: `/Users/minkyu/workspace/haetae/runs/v3/run.json`
- authoritative generation manifest: `/Users/minkyu/workspace/haetae/runs/v3/latest.json`
- derived progress: `/Users/minkyu/workspace/haetae/runs/v3/progress.json`

## Next actions

1. Commit and push the exact-review fixes, then request a targeted 6 Pro re-review of the new SHA.
2. Start the fresh 1,536-token `runs/v3` baseline after the targeted re-review and keep its manifest details in this file.
3. After completion, run certification across held-out sources, calibration, stress tests, and CPU/MPS latency measurements.
4. Send the exact final code commit and measured results to 6 Pro, implement supported findings, and rerun affected evidence.
