# Haetae research state

Status: active

## Objective

Build and evaluate a clean-room, local System One decision model that is structurally closer to Jev than ordinary NLI wrappers: one packed, non-generative forward pass per typed question; dynamic Choice, Noul, and Score outputs; calibrated probability evaluation; Korean support; and local serving through a Jev-compatible API.

Success means a reproducible checkpoint, evaluation on named held-out datasets, calibration and selective-risk evidence, permutation/packing stress tests, measured local latency, and an external 6 Pro code review of the final implementation and results.

## Repository and review thread

- Repository: https://github.com/pumpkinredbean/haetae
- Branch: `main`
- Checkpoint implementation review commit: `33006e1`
- Aside conversation: `Clean Room Jev Reproduction`
- Aside URL: https://chatgpt.com/c/6aaeba00-b670-83e8-9c29-3370b3c7945d
- Review model: ChatGPT 6 Pro
- Interaction method: Aside **REPL**, never Aside exec

## Confirmed design

- ModernBERT-base backbone.
- One row per question: `[CLS] state [SEP] instructions [SEP] option1 [SEP] option2 ...`.
- A shared scalar head reads the bidirectional hidden state of the separator preceding each option.
- Candidate logits are normalized only within their question.
- Choice order is augmented during training; Score order and Noul yes-first semantics remain fixed.
- Hard/soft cross-entropy plus Brier is the clean baseline. CE plus ranked probability score for Score is the first planned loss ablation.
- Training mixture has 13 sources, including KLUE-YNAT and NSMC for Korean.
- Calibration claims must name the dataset and deployment conditions. Selective action uses a one-sided binomial risk bound, not ordinary conformal coverage alone.

## 6 Pro findings already implemented

- Optimizer now covers all trainable backbone decay/no-decay parameters and head parameters exactly once.
- Candidate text cannot be silently truncated; the state yields token budget first.
- Loss inputs, option count, target shapes, label bounds, finiteness, and normalization are validated.
- Noul serving no longer splits `yes` into character candidates.
- Civil Comments hard-label orientation is fixed.
- HelpSteer2 groups all responses for the same prompt on one split.
- Temperature is fitted in log space; conformal rank and binomial boundary cases are fixed.
- Evaluation and serving explicitly handle rejected packs.
- Multi-question serving batches all questions into one backbone call, while still encoding the state once per batch row.

## Interruption finding

The first long run reached step 400/3000 but had no periodic checkpoint and was lost when the execution session disappeared. `runs/v1/model.pt` predates that run and must not be treated as its result.

Checkpoint-resume work is implemented in `src/haetae/train.py`: atomic `runs/v1/checkpoint.pt`, `runs/v1/progress.json`, model/optimizer/scheduler/config/RNG state, SIGTERM/SIGINT save, `--save-every 50`, and `--resume auto`.

Verified on the real training path:

- periodic checkpoint written at step 50;
- SIGTERM received during training and produced an atomic step-58 checkpoint;
- tmux session exited cleanly with `stopped with resumable checkpoint`;
- the same command was relaunched with `--resume auto` and logged `resumed runs/v1/checkpoint.pt at step 58`;
- training continued through a new periodic step-100 checkpoint, proving optimizer/scheduler state was usable after restart;
- an independent tiny-model round-trip restored weights, optimizer/scheduler state, step, skipped count, and progress metadata.

6 Pro then found that v1 restored RNG but not the current shuffled data order or batch cursor, so resumed optimization diverged from an uninterrupted run. The v1 baseline was stopped safely at step 368 and retained under `runs/v1` as a debugging artifact.

Checkpoint format v2 additionally stores and validates the epoch, full shuffled row order, next-batch cursor, training-row fingerprint, checkpoint version, required fields, step range, and scheduler `last_epoch == step`. Unit tests verify model/optimizer/scheduler/RNG/data-state restoration and reject a mismatched dataset fingerprint. The clean run uses `runs/v2`.

Real v2 recovery was verified: SIGTERM saved step 82 at epoch 0, cursor 656, scheduler step 82, and the next eight row indices. `--resume auto` loaded step 82 and the next step-100 checkpoint recorded cursor 800, exactly `656 + (100 - 82) * 8`, with the same training fingerprint. This confirms that the saved shuffled order and batch cursor were used after restart.

## State truncation audit

At `max_len=768`, 1,700 source rows per ordinary source and 1,700 HelpSteer2 source rows expanded to 8,500 questions were audited with the actual tokenizer and packing budget.

- 11 sources: 0 truncated states.
- BoolQ: 1/1,700 truncated (0.1%), with 82.0% of the state retained.
- HelpSteer2: 765/8,500 truncated (9.0%); no empty states; mean retained ratio 98.0%; worst retained ratio 34.4%.
- HelpSteer2 sensitivity: 3.71% truncated at 1,024 tokens, 0.29% at 1,536, and 0.06% at 2,048.

The current step-100 checkpoint remains a documented 768-token baseline. Before treating it as the final model, decide and review whether to keep and report prefix truncation, use segment-aware/head-tail preservation, or train a longer-context ablation.

## Long-running command

```bash
cd /Users/minkyu/workspace/haetae
tmux new-session -d -s haetae "HF_HUB_OFFLINE=1 uv run python -u -m haetae.train --sources ag_news,banking77,massive,mnli,anli,arc,emotion,klue_ynat,boolq,nsmc,civil_toxicity,sst5,helpsteer2 --per-source 1500 --eval-per-source 200 --steps 3000 --batch 8 --max-len 768 --out runs/v2 --save-every 50 --resume auto 2>&1 | tee -a train_v2.log"
```

- tmux session: `haetae`
- log: `/Users/minkyu/workspace/haetae/train_v2.log`
- checkpoint: `/Users/minkyu/workspace/haetae/runs/v2/checkpoint.pt`
- machine-readable progress: `/Users/minkyu/workspace/haetae/runs/v2/progress.json`

## Next actions

1. Read and reproduce the 6 Pro review of exact commit `33006e1`; fix supported findings and push a new SHA.
2. Ask 6 Pro to assess the measured HelpSteer2 truncation and choose a defensible baseline/ablation policy.
3. Keep the exact-resume v2 768-token baseline running in `runs/v2`; its real step-82 to step-100 cursor continuity is verified.
4. After step 3000, run `haetae.certify` across held-out sources, fix any harness errors, and record results.
5. Measure single-request and batched CPU/MPS latency and run permutation/option perturbation stress tests.
6. Send code and measured results to 6 Pro for final review; implement supported findings and rerun affected checks.
