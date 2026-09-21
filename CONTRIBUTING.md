# Contributing

## Before opening a change

Open an issue that states the problem, affected component, intended evidence,
and whether the change runs model inference or optimizer updates. Read
`docs/research-status.md`, `docs/evidence.md`, and `docs/licenses.md` first.

## Development setup

```bash
uv sync --frozen
PYTHONPATH=. uv run --with pytest pytest -q
uv run --with ruff ruff check .
```

`pytest` and `ruff` are explicit ephemeral development tools in these commands;
they are not runtime dependencies in the current historical package metadata.

Run focused tests while developing. Run the full available suite before a
pull request when the locked historical environment is installed. Tests must
not fetch or open a locked evaluation split.

## Research changes

- Freeze the protocol, population, metrics, stopping rule, and compute bound
  before inspecting a new result.
- Keep checkpoints, raw predictions, private path maps, and dataset rows out of
  Git.
- Preserve qualified artifacts. Write a new version instead of overwriting.
- Bind code, data, run, checkpoint, calibration, and split identities.
- Label exploratory work clearly and cite evidence IDs beside numeric claims.
- Do not treat development results as locked-test results.
- Do not change historical producer code to make an old result easier to
  reproduce. Add a versioned producer or runtime instead.

## Code changes

Keep modules small, typed where practical, and explicit about failure. Reject
partial candidate sets, stale artifacts, path escapes, and ambiguous schemas.
Add tests for behavior that protects evidence or model correctness. Avoid tests
that only repeat implementation details.

## Documentation and licensing

Every metric needs a registered evidence ID and enough context to interpret
the population and protocol. Keep repository code, model-weight, and dataset
licenses separate. Add a source to `docs/licenses.md` before using it in a
publishable model.

By submitting a contribution, you license it under Apache-2.0 and confirm that
you have the right to do so.
