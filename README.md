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
has not been evaluated on locked tests. The historical separate-question
ModernBERT model remains in the repository as the comparison baseline.

## Release status

| Item | Current state |
| --- | --- |
| Current model | shared-v1, generation 79 |
| Parameters | 140,593,792 total; 98,560 in the pointer head |
| Weight format | safetensors release candidate, float32 |
| Weight size | 562,389,888 bytes; tokenizer and config add about 34 MB |
| Training checkpoint | 1,687,348,267 bytes including optimizer and recovery state |
| Runtime | Local CLI, Python runtime, `/v1/decide`, and `/v1/systemone` API |
| Public weights | Not published while training-source redistribution terms remain unresolved |
| Locked-test evaluation | Not performed |

The code is open source under Apache-2.0. The current model weights are not an
open-weight release. A byte-verified local bundle exists for release testing,
but it is deliberately kept outside Git and must not be published until the
license ledger and runtime parity gate pass.

## Measured status

All differences below are shared-v1 minus the historical baseline after each
model's independently fitted temperature. This first table is common-clean and
question-weighted. Confidence intervals resample whole request parents within
the frozen strata.

| Public development population | Questions | Accuracy difference | NLL difference | Evidence |
| --- | ---: | ---: | ---: | --- |
| Decision | 1,463 | +0.1722, 95% CI [0.1389, 0.2047] | -0.4562, 95% CI [-0.5203, -0.3911] | `development_comparison` |
| Korean | 5,000 | +0.3866, 95% CI [0.3686, 0.4038] | -0.5028, 95% CI [-0.5317, -0.4738] | `development_comparison` |
| English transfer | 764 | -0.0563, 95% CI [-0.0976, -0.0118] | +0.1302, 95% CI [0.0810, 0.1775] | `development_comparison` |

The corresponding source-macro accuracy differences give every task source
equal weight:

| Public development population | Source-macro accuracy difference | Evidence |
| --- | ---: | --- |
| Decision | +0.0998, 95% CI [0.0686, 0.1322] | `development_comparison` |
| Korean | +0.3881, 95% CI [0.3684, 0.4073] | `development_comparison` |
| English transfer | -0.0228, 95% CI [-0.0570, 0.0126] | `development_comparison` |

These intervals are pointwise and conditional on the fixed checkpoints,
fitted temperatures, observed parents, and declared strata. They do not cover
training-seed or calibration-fit uncertainty.

On the frozen six-process, warmed MPS batch-one benchmark, packed execution
used 0.7634 times the matched batched latency for decision workloads and
0.8364 times it for Korean workloads. The one-question negative control ratio
was 0.9881. These ratios apply only to the recorded Apple M3 Pro setup and do
not describe cold start, transport, or peak memory. Evidence:
`confirmation_summary`.

The single-question control is pooled-neutral under exact counterbalancing but
has a strong immediate-repeat position effect: its order-stratum ratios were
0.5568 and 1.7538, and the first operation took 1.7748 times the second on
average. Both primary order strata and all six processes favored packed
execution. The accepted claim is limited to the frozen warmed, adjacent-pair
workload on the recorded host. Evidence: `confirmation_summary`.

The transfer regression is a current limitation. A cached audit also found no
shared-v1 training supervision for the emotion and offensive-tweet tasks. The
producer is commit `35ee1327b6d53a722d953f067e4df081f88ee6c7`;
the corrected exploratory audit archive has SHA-256
`ad503e2520368393f1533a3ddbb41522750ed40afc988fd932d41d0df48b0990`
and manifest content SHA-256
`dacbfd8061eed3623e39f6b3413748bf4a2183e16318df07fb51fc344abbc11c`.
The next frozen experiment tests whether the failures come from task/readout
binding before any additional training is considered.

See [the current research status](docs/research-status.md),
[model card](models/shared-v1/README.md), and
[evidence guide](docs/evidence.md) for the scope of these claims.

## Local runtime

Install the runtime and optional HTTP server dependencies:

```bash
uv sync --extra serve
```

The bundle path must contain the exact pinned `model.safetensors`, tokenizer,
configuration, and manifest. Run one local decision from JSON:

```bash
uv run haetae decide \
  --bundle /path/to/shared-v1-bundle \
  --input examples/shared-v1-request.json
```

Or start the local API:

```bash
uv run haetae serve \
  --bundle /path/to/shared-v1-bundle \
  --host 127.0.0.1 \
  --port 8000
```

`/v1/decide` accepts the explicit list form in
[`examples/shared-v1-request.json`](examples/shared-v1-request.json).
`/v1/systemone` accepts the compatibility form below.

## Interface

A request contains one state and typed questions:

```json
{
  "state": "The customer says the transfer has not arrived.",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Does this require urgent handling?"
    },
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "payments": null,
        "account_access": null,
        "general_support": null
      }
    }
  }
}
```

`noul` is a two-option decision with `yes` at index 0. `score` options have a
declared low-to-high order. Candidate text is never silently truncated: a
request that cannot preserve every option within the input budget is rejected.

This compatibility endpoint is part of the shared-v1 release-candidate
runtime. It returns uncalibrated probabilities and reports
`calibrated: false`. It does not make shared-v1 a deployed service.

## Repository map

```text
src/haetae/shared_v1.py     versioned shared-v1 architecture
src/haetae/runtime.py       verified local bundle loader and inference
src/haetae/api.py           explicit and /v1/systemone HTTP interfaces
src/haetae/cli.py           local decide and serve commands
src/haetae/                 historical baseline and research utilities
experiments/shared_state.py shared-state model implementation
experiments/train_shared.py immutable shared-v1 training producer
experiments/evaluate_*.py   frozen public-development evaluators
experiments/benchmark_*.py  execution benchmarks
research/evidence/          hash-only evidence registry
tools/evidence_registry.py  local evidence verifier
docs/                       research, evidence, reproduction, and licenses
```

## Reproducing the research

The environment is locked by `uv.lock`. Runtime dependencies are installed by
default; research dependencies are explicit extras:

```bash
uv sync --frozen --extra research --extra serve --extra dev
uv run python tools/evidence_registry.py self-test
```

Exact artifact verification also needs a local path map because checkpoints
and measured prediction files are intentionally not stored in Git:

```bash
uv run python tools/evidence_registry.py verify \
  --paths /path/to/local-evidence-paths.json \
  --out /path/to/verification-report.json
```

The verifier checks content digests, sizes, provenance chains, source-code
identity, and registered `contains_locked_data: false` attestations. It does
not inspect arbitrary payloads to independently establish that they contain no
locked-test examples. See [the reproduction guide](docs/reproduction.md)
before rerunning any producer.

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
