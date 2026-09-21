# Current research status

## Classification

Haetae shared-v1 is a validated research prototype. Generation 79 is the
current checkpoint. The repository does not claim equivalence to a proprietary
system, recovery of a private training method, production readiness, universal
latency, or calibrated probabilities for arbitrary inputs.

## Fixed model

- Backbone: `jhu-clsp/mmBERT-small`
- Backbone revision: `abc32620dd4f6ab06f5fbe905dc25f310618e09f`
- Parameters: 140,593,792
- Float32 safetensors size: 562,389,888 bytes
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
advantage when pooled under exact counterbalancing. It did show a strong
immediate-repeat position effect, with order-stratum ratios 0.5568 and 1.7538.
Both primary order strata and all six processes favored packed execution. This
result is limited to the frozen warmed, adjacent-pair workload on the recorded
Apple M3 Pro environment.
Evidence: `confirmation_protocol`, `confirmation_summary`, and
`confirmation_measured_archive`.

## Active scientific question

The transfer loss is not yet evidence that the architecture lacks useful
features. A cached audit found no shared-v1 training supervision for the
emotion or offensive-tweet tasks. On offensive tweets, shared-v1 had worse
accuracy and NLL but better cached AUROC than the historical baseline. That
pattern is consistent with a threshold or task-binding problem and does not
prove one.

The cached audit is exploratory evidence produced by commit
`35ee1327b6d53a722d953f067e4df081f88ee6c7`. Its corrected archive SHA-256 is
`ad503e2520368393f1533a3ddbb41522750ed40afc988fd932d41d0df48b0990`,
and its manifest content SHA-256 is
`dacbfd8061eed3623e39f6b3413748bf4a2183e16318df07fb51fc344abbc11c`.

The fixed-weight diagnostic and its corrected semantic-output replay are
complete. Explicit native task descriptions improved the emotion and
offensive development results, but offensive recall fell sharply and wording
and option-order sensitivity remained substantial. The evidence supports a
bounded supervision experiment rather than a prompt-only repair.

The paired replay implementation and all four deterministic draw tapes are
frozen on `next/science-coverage`. No new weights have been produced. The M4
pre-run review returned a hold after reproducing seven integration and
execution-guard defects. The science branch must correct and re-freeze those
boundaries before another review; neither seed-17 arm is authorized.

## Release blockers

1. Finish source-specific licensing review before distributing generation 79
   weights.
2. Complete the 32-request parity review between the exported data-only bundle
   and the original generation 79 checkpoint.
3. Publish only aggregate qualified evidence. Locked tests remain unopened.

The portable runtime and HTTP extras have passed a clean-wheel install and
CPU smoke test. Continuous integration repeats the repository tests, release
surface checks, wheel build, isolated install, CLI import, and HTTP-extra
import without downloading model weights.
