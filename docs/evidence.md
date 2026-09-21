# Evidence guide

Haetae separates claims from artifact locations. Public documents cite stable
evidence IDs. `research/evidence/index.json` binds each ID to a SHA-256 digest,
byte size, claim scope, and provenance. Local checkpoint and prediction paths
remain outside Git.

## Main claim map

| Evidence ID | Claim scope |
| --- | --- |
| `shared_checkpoint_g79` | Completed generation 79 weights |
| `shared_run_spec` | Immutable model, optimizer, data, and source-code identity |
| `shared_run_manifest` | Completed checkpoint generation binding |
| `shared_suite_manifest` | Frozen train, calibration, and development membership |
| `shared_suite_rendered_state_audit` | Rendered-state overlap and binding audit |
| `comparison_plan` | Common-clean public-development membership |
| `development_comparison` | Paired shared-v1 versus baseline metrics |
| `execution_protocol` | Frozen numerical and execution protocol |
| `execution_summary` | Numerical equivalence, timing, and memory result |
| `confirmation_protocol` | Frozen six-process MPS confirmation protocol |
| `confirmation_summary` | Qualified warmed MPS batch-one result |

The registry contains additional IDs for the historical baseline, frozen
development populations, and externally produced comparison studies. An ID in
the registry means that its bytes and claim scope are recorded; it does not
turn exploratory work into a qualified Haetae result.

## Verification

Run the registry's adversarial self-tests without any external artifacts:

```bash
uv run python tools/evidence_registry.py self-test
```

To verify the actual evidence, create a complete local path map outside the
repository. Every registered evidence ID needs an `artifacts` mapping, even
when the registry marks that artifact optional. Each relative path must stay
inside a root's allowlist. A minimal synthetic shape is:

```json
{
  "schema_version": 1,
  "reference_repository": {
    "path": "/absolute/path/to/detached-reference"
  },
  "roots": {
    "evidence": {
      "path": "/absolute/path/to/artifacts",
      "allow_prefixes": ["runs/shared-v1", "evaluations"]
    }
  },
  "artifacts": {
    "shared_run_spec": {
      "root": "evidence",
      "relative_path": "runs/shared-v1/run.json"
    }
  }
}
```

The example shows one entry for readability; a working map must include every
ID in `research/evidence/index.json`. The reference checkout must be detached
at commit `4828f9c69ba0f5b1f64fd72aa35fa3ce7269d294`. Its registered source files
and their parent directories must be nonwritable. Then run:

```bash
uv run python tools/evidence_registry.py verify \
  --registry research/evidence/index.json \
  --paths /path/to/local-evidence-paths.json \
  --out /path/to/verification-report.json
```

The verifier refuses duplicate JSON keys, private paths in public metadata,
symbolic-link escapes, digest or size mismatches, stale derivative ancestors,
locked-test entries, mutable reference sources, and output replacement. It
checks every derivative against the same captured bytes used for parsing and
hashing.

## Evidence rules

- Publish the protocol before inference or training.
- Record code, data, run, checkpoint, and split identities.
- Keep locked-test bytes out of the evidence boundary.
- Preserve qualified artifacts; publish a new version instead of overwriting.
- Label exploratory results as exploratory.
- Cite an evidence ID beside every numeric claim.
- State the population, hardware, calibration, and uncertainty scope needed to
  interpret a metric.

The registry is digest-bound, not cryptographically signed. Integrity depends
on preserving the Git history, registered bytes, and local path map together.
