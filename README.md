# Haetae

Haetae is a local research model for typed decisions. It accepts a shared
state and one or more `choice`, `noul`, or `score` questions, then returns a
probability distribution over each question's options without generating
text.

The current model is **shared-v1, checkpoint generation 79**. It uses the
multilingual `jhu-clsp/mmBERT-small` backbone at revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`. The encoder reads the state once;
each question has an isolated bidirectional branch that can attend to the
state and itself. Sibling questions cannot attend to one another, and state
tokens cannot attend back to question branches. A listwise pointer head scores
the options in each branch.

This repository is a research alpha. The checkpoint is validated on frozen
public development populations, but it is not a production safety system and
has not been evaluated on locked tests. A portable inference bundle is still
being prepared. The historical separate-question ModernBERT model remains in
the repository as the comparison baseline.

## Measured status

All differences below are shared-v1 minus the historical baseline after each
model's independently fitted temperature. Confidence intervals resample whole
request parents.

| Public development population | Questions | Accuracy difference | NLL difference | Evidence |
| --- | ---: | ---: | ---: | --- |
| Decision | 1,463 | +0.1722, 95% CI [0.1389, 0.2047] | -0.4562, 95% CI [-0.5203, -0.3911] | `development_comparison` |
| Korean | 5,000 | +0.3866, 95% CI [0.3686, 0.4038] | -0.5028, 95% CI [-0.5317, -0.4738] | `development_comparison` |
| English transfer | 764 | -0.0563, 95% CI [-0.0976, -0.0118] | +0.1302, 95% CI [0.0810, 0.1775] | `development_comparison` |

On the frozen six-process, warmed MPS batch-one benchmark, packed execution
used 0.7634 times the matched batched latency for decision workloads and
0.8364 times it for Korean workloads. The one-question negative control ratio
was 0.9881. These ratios apply only to the recorded Apple M3 Pro setup and do
not describe cold start, transport, or peak memory. Evidence:
`confirmation_summary`.

The transfer regression is a current limitation. A cached audit also found no
shared-v1 training supervision for the emotion and offensive-tweet tasks. The
next frozen experiment tests whether their failures come from task/readout
binding before any additional training is considered.

See [the current research status](docs/research-status.md),
[model card](models/shared-v1/README.md), and
[evidence guide](docs/evidence.md) for the scope of these claims.

## Interface

A request contains one state and typed questions:

```json
{
  "state": "The customer says the transfer has not arrived.",
  "nouls": [
    {
      "id": "urgent",
      "instructions": "Does this require urgent handling?"
    }
  ],
  "choices": [
    {
      "id": "route",
      "instructions": "Which team should handle this?",
      "options": ["payments", "account access", "general support"]
    }
  ],
  "scores": []
}
```

`noul` is a two-option decision with `yes` at index 0. `score` options have a
declared low-to-high order. Candidate text is never silently truncated: a
request that cannot preserve every option within the input budget is rejected.

The current checked-in server loads the historical baseline. Do not use it as
an example of shared-v1 deployment. The shared-v1 portable runtime will be
documented after export and parity review.

## Repository map

```text
src/haetae/                 historical baseline, calibration, and server
experiments/shared_state.py shared-state model implementation
experiments/train_shared.py immutable shared-v1 training producer
experiments/evaluate_*.py   frozen public-development evaluators
experiments/benchmark_*.py  execution benchmarks
research/evidence/          hash-only evidence registry
tools/evidence_registry.py  local evidence verifier
docs/                       research, evidence, reproduction, and licenses
```

## Reproducing the research

The historical producer environment is locked by `uv.lock`:

```bash
uv sync
uv run python tools/evidence_registry.py self-test
```

Exact artifact verification also needs a local path map because checkpoints
and measured prediction files are intentionally not stored in Git:

```bash
uv run python tools/evidence_registry.py verify \
  --paths /path/to/local-evidence-paths.json \
  --out /path/to/verification-report.json
```

The verifier checks content digests, sizes, provenance chains, the source-code
identity, and the absence of locked-test artifacts. See
[the reproduction guide](docs/reproduction.md) before rerunning any producer.

## Licensing

Repository code and documentation are available under Apache-2.0. That license
does not grant rights to third-party datasets or to Haetae model weights.
Generation 79 weights are not distributed by this research-alpha repository
while source-specific redistribution terms remain unresolved. See
[the license ledger](docs/licenses.md) and [NOTICE](NOTICE).

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing research producers or
publishing a metric. Report security issues through GitHub's private
vulnerability reporting channel as described in [SECURITY.md](SECURITY.md).
