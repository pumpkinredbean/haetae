# Shared-state experiment

The active baseline repeats the state for every question. This prototype
stores the state once and appends isolated question branches. A branch can
attend to the state and itself; state tokens cannot attend back to branches,
and sibling branches cannot attend to each other. Logical branch positions
restart after the state. A pointer head compares each option representation
with a decision token.

This structure tests the strongest public reconstruction of Jev's observable
behavior without claiming access to Jev's private architecture. It follows the
shared-state and sibling-isolation evidence described in Archer Hume's
[black-box study](https://archerhume.com/posts/jevs-architecture-unmasked) and
is compared against the Apache-2.0
[`kev`](https://github.com/jaredpalmer/kev) implementation at commit
`20fa6268c8ceb226530be2fb5266ab2c36b37724`.

The proposed backbone is
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) at
revision `abc32620dd4f6ab06f5fbe905dc25f310618e09f`. It has 140 million
parameters, an 8,192-token limit, and multilingual pretraining that includes
Korean. The model card and [paper](https://arxiv.org/abs/2509.06888) report
pretraining over more than 1,800 languages. This keeps the experiment close to
the baseline's size while correcting the baseline's English-only pretraining.

The prototype currently proves these mechanics:

- packed questions match separate execution within numerical attention-kernel
  tolerance;
- changing one question cannot change a sibling's result;
- state hidden states cannot receive information from a question;
- state tokens are stored once, and state truncation never removes question or
  option content;
- full and sliding attention use logical branch positions, so later packed
  branches have the same receptive field as a single branch.

A real CPU probe with the pinned mmBERT-small checkpoint packed two Korean
questions into 73 tokens instead of 42 and 54 tokens in two separate passes.
The maximum packed-versus-separate logit differences were `1.67e-6` and
`1.03e-5` before training.

The frozen Kev decision-v4 suite was inspected without opening its locked test
partition. Its manifest SHA-256 is
`1b33e566d114f9eafeff55b36c221fadb2a4ae358a1b9cc68006e82c7cfad8f1`.
The training partition SHA-256 is
`cb55b79e037c9f5a0ef2de4efe79ec6ef5d7d8d21aa64e94eee7731d7eddce74`.
It contains 10,896 requests and 13,896 questions; 2,000 requests contain two
or three questions over one state. All requests fit the pinned mmBERT
tokenizer under the 2,048-token policy without state truncation. The longest
packed request is 1,011 tokens.

The transfer-v4 manifest SHA-256 is
`31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1`,
and its development partition SHA-256 is
`ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e`.
It contains 764 single-question requests. It can measure transfer quality but
cannot measure the multi-question state-sharing speedup by itself.

The experiment must pass the following gates before a locked test is read:

1. preserve sibling isolation and complete option coverage on every evaluated
   request;
2. compare the current baseline and the shared-state model on the same frozen
   `kev` transfer development bytes;
3. freeze a Korean transfer suite with parent-disjoint development and locked
   roles;
4. improve transfer NLL or Brier without a material regression in accuracy;
5. report memory and latency for one question and mixed multi-question
   requests on an idle host;
6. predeclare the promotion rule and run each recipe across multiple seeds.

The shared-state trainer accepts only manifest-declared bytes and refuses the
locked test unless a caller passes an explicit gate. A typical first seed is:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.train_shared \
  --suite /path/to/kev/evals/v4/decision-v4 \
  --steps 1500 --batch 8 --microbatch 2 --max-len 2048 \
  --gradient-checkpointing --local-files-only \
  --out runs/shared-mmbert-s17 --seed 17 --resume none
```

The trainer publishes immutable periodic generations with the exact suite,
backbone, tokenizer, code, optimizer, scheduler, data order, and random state.
An interrupted run resumes only when all bound identities still match.

The active baseline run is independent of this directory. Prototype changes do
not alter its training-code identity.

## Trained-weight execution benchmark

`benchmark_shared_execution.py` compares packed requests, batched
single-question encodings, and serial single-question encodings without
changing model weights. The checked-in protocol fixes numerical gates,
result-independent workloads, latency repetitions, memory checks, and the
next decision before inference.

Freeze the public development population after committing the runner:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.benchmark_shared_execution freeze \
  --run runs/shared-v1 \
  --comparison-plan evaluations/comparison-v1/plan.json \
  --decision-suite evaluations/shared-v1 \
  --transfer-suite evaluations/kev-transfer-v4 \
  --korean-suite evaluations/korean-v1 \
  --out evaluations/shared-execution-v1
```

Run `equivalence` once per device, `timing` for repetitions 0 through 2 on
each device, and `memory` for arms P, B, and S on each device. `summarize`
validates every artifact binding and applies the frozen gates. These commands
load development files only; they have no option that permits a locked test.
