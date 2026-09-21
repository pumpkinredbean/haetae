# Current research status

## Classification

Haetae shared-v1 is a validated research prototype. Generation 79 is the
current checkpoint. The repository does not claim equivalence to a proprietary
system, recovery of a private training method, production readiness, universal
latency, or calibrated probabilities for arbitrary inputs.

## Fixed model

- Backbone: `jhu-clsp/mmBERT-small`
- Backbone revision: `abc32620dd4f6ab06f5fbe905dc25f310618e09f`
- Checkpoint generation: 79
- Checkpoint SHA-256: `e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c`
- Training: 3,894 optimizer steps, two request epochs
- Frozen train population: 15,572 requests and 23,068 questions
- Maximum training request length: 1,011 tokens under a 2,048-token policy
- Locked test opened: no

Evidence IDs: `shared_run_spec`, `shared_run_manifest`,
`shared_checkpoint_g79`, `shared_suite_manifest`, and `shared_suite_train`.

## What has been established

The shared-state implementation passed structural checks for sibling
isolation, one-way state visibility, restarted logical positions, and complete
candidate preservation. Packed, batched separate-question, and serial
separate-question execution produced equivalent decisions under the frozen
numerical protocol. Evidence: `execution_protocol` and `execution_summary`.

On common-clean public development data, shared-v1 improved decision and
Korean accuracy, NLL, and Brier metrics relative to the historical baseline.
It regressed on aggregate English transfer accuracy, NLL, and Brier metrics.
The paired comparison uses whole request parents and independently fitted
temperatures. Evidence: `comparison_plan`, `baseline_development_report`,
`shared_development_report`, and `development_comparison`.

The warmed MPS batch-one confirmation used six fresh processes. All six
processes showed lower packed latency on both primary multi-question
populations. The one-question negative control did not show a material
advantage. This result is specific to the recorded Apple M3 Pro environment.
Evidence: `confirmation_protocol`, `confirmation_summary`, and
`confirmation_measured_archive`.

## Open scientific question

The transfer loss is not yet evidence that the architecture lacks useful
features. A cached audit found no shared-v1 training supervision for the
emotion or offensive-tweet tasks. On offensive tweets, shared-v1 had worse
accuracy and NLL but better cached AUROC than the historical baseline. That
pattern is consistent with a threshold or task-binding problem and does not
prove one.

The next experiment is a frozen, fixed-weight diagnostic with no optimizer
updates. It tests native labels, cyclic and opaque label mappings, ranking,
thresholds, calibration, and fixed logistic probes on bounded public support
sets. New inference is capped at 20,000 encoded sequences. Paired training
replay is allowed only if those diagnostics justify it.

## Release blockers

1. Finish source-specific licensing review before distributing generation 79
   weights.
2. Export a data-only checkpoint and prove parity with the original generation
   79 checkpoint.
3. Add an offline portable runtime and clean-install package tests.
4. Finish the fixed-weight diagnostic and record the scientific decision,
   including a negative result if that is what the evidence supports.
5. Publish only aggregate qualified evidence. Locked tests remain unopened.

