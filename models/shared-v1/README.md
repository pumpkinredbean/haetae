# Haetae shared-v1

## Model description

Haetae shared-v1 is an encoder-only model for typed decisions over a shared
state. It returns one probability distribution per question and generates no
text.

- Model version: shared-v1, generation 79
- Backbone: `jhu-clsp/mmBERT-small`
- Backbone revision: `abc32620dd4f6ab06f5fbe905dc25f310618e09f`
- Backbone license metadata: MIT
- Languages represented in supervised training: English and Korean
- Input primitives: `choice`, `noul`, and ordinal `score`
- Maximum trained packed length: 2,048 tokens
- Checkpoint SHA-256: `e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c`

The model encodes state tokens once. Each question branch can attend to the
state and to tokens in the same branch. State tokens cannot attend to a
question, and sibling questions cannot attend to one another. Logical branch
positions restart after the state. A pointer head compares contextual option
representations within each question.

Evidence: `shared_run_spec`, `shared_run_manifest`, and
`shared_checkpoint_g79`.

## Intended use

The model is intended for local research on routing, classification, ordinal
scoring, shared-context execution, calibration, and selective decision
policies. Suitable uses require an application-specific development set,
explicit option semantics, monitoring, and human review for consequential
decisions.

It is not validated for autonomous high-impact decisions, open-ended factual
generation, safety moderation without task-specific evaluation, or inputs that
require knowledge beyond the frozen training and development evidence.

## Training

Generation 79 was trained for 3,894 optimizer steps, corresponding to two
epochs over 15,572 requests and 23,068 questions. The request objective first
averages question losses within each request and then averages requests. The
loss combines cross-entropy, multiclass Brier loss, and an ordinal term for
ordered score questions. Training used English public classification and
reasoning sources, Korean news and sentiment sources, and synthetic policy and
composition sources.

Evidence: `shared_run_spec`, `shared_suite_manifest`, `shared_suite_train`, and
`shared_suite_rendered_state_audit`.

## Evaluation

All reported metrics use frozen public development populations. No locked test
was opened. The comparison below is shared-v1 minus the historical baseline
after independently fitting a temperature for each model.

| Population | Accuracy difference | NLL difference | Interpretation |
| --- | ---: | ---: | --- |
| Decision, 1,463 questions | +0.1722 | -0.4562 | shared-v1 better |
| Korean, 5,000 questions | +0.3866 | -0.5028 | shared-v1 better |
| English transfer, 764 questions | -0.0563 | +0.1302 | shared-v1 worse |

Evidence: `development_comparison`. Confidence intervals and source-macro
results are recorded in that artifact and summarized in the repository
README.

The warmed six-process MPS batch-one confirmation measured packed-to-batched
latency ratios of 0.7634 on decision workloads and 0.8364 on Korean workloads.
The one-question negative control ratio was 0.9881. These are host- and
protocol-specific results. Evidence: `confirmation_protocol` and
`confirmation_summary`.

## Probability interpretation

Outputs are softmax distributions over the options supplied for one question.
The development evaluator fits temperatures on frozen calibration data. That
does not establish calibration on arbitrary tasks, domains, languages, option
sets, or deployment populations. The model card therefore makes no general
claim that its probabilities are calibrated.

Conformal prediction sets and selective-risk bounds are separate procedures
in the historical baseline research code. Their guarantees apply only when
their stated exchangeability, population, and artifact bindings hold. They do
not replace probability calibration.

## Limitations

- Aggregate English transfer results are worse than the historical baseline.
- The emotion and offensive-tweet development tasks had no matching shared-v1
  training supervision in the cached coverage audit.
- Option wording, order, and label semantics can change predictions.
- Long requests are rejected if all candidates cannot be preserved; they are
  not silently truncated.
- Exact semantic decontamination from backbone pretraining is not certified.
- Reported latency does not include cold start, network transport, or a
  production service stack.
- Evaluation covers English and Korean development data and does not establish
  comparable quality in other languages present in the backbone.

## Licensing and distribution

The mmBERT-small model card declares the backbone under MIT. Haetae's source
code is Apache-2.0. Those licenses do not determine the terms of third-party
training datasets. Generation 79 weights are not distributed with the current
research alpha because several source licenses or redistribution terms remain
unresolved. See `docs/licenses.md` before publishing any bundle.

