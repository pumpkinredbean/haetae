# License and provenance ledger

Status date: 2026-09-21.

This ledger separates repository code, backbone weights, Haetae weights, and
dataset terms. A license for one layer does not grant rights to another layer.

## Distribution decision

| Layer | Current status |
| --- | --- |
| Haetae code and documentation | Apache-2.0 |
| Historical ModernBERT-base backbone | Upstream card declares Apache-2.0 |
| Current mmBERT-small backbone | Upstream card declares MIT |
| Haetae shared-v1 generation 79 weights | Not distributed; source-specific terms are unresolved |
| Frozen data and prediction artifacts | Not distributed from this repository |
| Aggregate qualified metrics | Eligible for publication after evidence review |

The project license applies only to files for which Haetae contributors can
grant rights. It does not relicense a model checkpoint or dataset.

## Backbone provenance

| Component | Pinned revision | Declared license | Primary source |
| --- | --- | --- | --- |
| `jhu-clsp/mmBERT-small` | `abc32620dd4f6ab06f5fbe905dc25f310618e09f` | MIT | [revision card](https://huggingface.co/jhu-clsp/mmBERT-small/tree/abc32620dd4f6ab06f5fbe905dc25f310618e09f) |
| `answerdotai/ModernBERT-base` | historical baseline identity in `baseline_run_spec` | Apache-2.0 | [model card](https://huggingface.co/answerdotai/ModernBERT-base) |

The backbone license is necessary but not sufficient for distributing a
fine-tuned checkpoint. Training-data terms remain a separate decision.

## Generation 79 supervised sources

The exact data membership is bound by `shared_suite_manifest` and
`shared_suite_train`. The table records license metadata found in the pinned
source or official upstream repository. `Unresolved` means that the cited
source did not state a single clear redistribution license, used `unknown` or
`other`, or combined sources with different terms.

| Training source | Pinned source revision | Recorded terms | Weight-release status |
| --- | --- | --- | --- |
| TREC | `CogComp/trec@65752bf53af25bc935a0dce92fb5b6c930728450` | No license in the pinned dataset card | Unresolved |
| Banking77 | `legacy-datasets/banking77@f54121560de48f2852f90be299010d1d6dc612ec` | CC BY 4.0 metadata | Attribution required |
| IMDb | `stanfordnlp/imdb@e6281661ce1c48d982bc483cf8a173c1bbeb5d31` | `other` metadata | Unresolved |
| AG News | `fancyzhx/ag_news@eb185aade064a813bc0b7f42de02595523103ca4` | `unknown` metadata | Unresolved |
| Amazon Reviews Multi | `SetFit/amazon_reviews_multi_en@ec73b665e4be0f567b69d39425355401cfe0d29b` | Apache-2.0 metadata | Notice required |
| DBpedia 14 | `fancyzhx/dbpedia_14@9abd46cf7fc8b4c64290f26993c540b92aa145ac` | CC BY-SA 3.0 metadata | Share-alike analysis required |
| BoolQ | `google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5` | CC BY-SA 3.0 metadata | Share-alike analysis required |
| SST-5 | `SetFit/sst5@e51bdcd8cd3a30da231967c1a249ba59361279a3` | No license in the pinned dataset card | Unresolved |
| MultiNLI | `nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221` | Mixed CC BY, CC BY-SA, MIT, and `other` metadata | Per-source review required |
| Yelp Review Full | `Yelp/yelp_review_full@c1f9ee939b7d05667af864ee1cb066393154bf85` | `other` metadata | Unresolved |
| KLUE YNAT | source files recorded in `korean_suite_manifest` | CC BY-SA 4.0 in the [official repository](https://github.com/KLUE-benchmark/KLUE#license) | Share-alike analysis required |
| NSMC | source file recorded in `korean_suite_manifest` | No license grant in the [official repository](https://github.com/e9t/nsmc) | Unresolved |
| Synthetic policy and composition | `jaredpalmer/kev-suites@57a3ffd3951432855c96eceeef4362617f6a057d` | Dataset bundle has no declared license; generator code is Apache-2.0 | Unresolved |

Official Hugging Face metadata for the pinned revisions is available from each
dataset's revision page. Metadata describes the publisher's declaration; this
project does not infer extra permissions from a repository being publicly
downloadable.

## Diagnostic-only sources

These sources are not part of generation 79 training. They are listed because
the frozen next experiment may use bounded public support and development
splits.

| Source | Recorded terms | Policy |
| --- | --- | --- |
| `dair-ai/emotion` | Official card metadata says `other` | Do not redistribute rows; record aggregate diagnostics only |
| `cardiffnlp/tweet_eval`, offensive subset | Official card says subset licenses vary and the offensive subset is undefined; platform terms also apply | Do not redistribute tweet text or derived row-level artifacts |

## Conditions for publishing model weights

Generation 79 remains private until all of the following are recorded in a
reviewed release commit:

1. each unresolved training source has an upstream license or a documented
   exclusion and clean replacement;
2. attribution and share-alike obligations are mapped to the planned model
   distribution;
3. the exported bundle includes the mmBERT-small MIT notice and all required
   dataset notices;
4. the bundle contains no raw dataset rows, optimizer state, random state, or
   private paths;
5. the model card and distribution license state exactly which files each
   license covers.

Until those conditions pass, references to generation 79 identify internal
evidence and do not offer checkpoint bytes for redistribution.

