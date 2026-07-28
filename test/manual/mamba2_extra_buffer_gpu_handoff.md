# Mamba2 extra_buffer SSM Track-State GPU Handoff

## Context

Branch: `fix-track-ssm-offset`

Code commit: `6af40ed722ef6a0d912cfd802a25aef5bfb84836`

Fix target: upstream issue `sgl-project/sglang#31833`.

Upstream status checked on 2026-07-28:

- Issue `#31833` is still open.
- Upstream `main` still uses the old `_init_track_ssm_indices` path.
- PR `#32555` is related but distinct. It fixes a decode CUDA-graph
  track-buffer bug, not this multi-request prefill Mamba2 SSM tracking bug.

## Fix Summary

This patch fixes Mamba2 `extra_buffer` radix-cache tracking for multi-request
prefill batches where a request's track boundary is not also a flattened
physical Mamba2 chunk boundary.

Behavior:

- Non-Mamba2 / FLA-style backends keep the existing per-request `h` indexing
  path.
- Mamba2 extend tracking records flattened `[start, end)` token bounds for
  non-final tracked states.
- `MambaMixer2` reconstructs exact tracked SSM states from scan intermediates
  with `chunk_state_varlen_by_bounds`.
- Final tracked Mamba2 states still copy from the already-updated cache state.
- Mamba2 backend no longer copies extend track state from `intermediate_states`,
  avoiding overwrite with the old approximate state.

## Non-GPU Checks Already Run

Passed:

- `git diff --check`
- `python3 -m compileall -q ...` on all changed Python files
- `python3 -m ruff check --select F401,F821,F822,F823,F841 ...` on all changed
  Python files
- `SKIP=rustfmt-sgl-model-gateway,rustfmt-sgl-router PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files`

Notes:

- Full `PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files` ran
  all non-Rust hooks but failed only because this machine has no `rustup` or
  `cargo`.
- `pre-commit` reformatted/import-sorted the staged Python changes, then the
  skipped-Rust rerun passed.
- Local pytest could not run because this machine does not have `torch`
  installed: `ModuleNotFoundError: No module named 'torch'`.

## Run On GPU Machine

After pulling the branch:

```bash
git log --oneline -2
git show --stat --oneline --no-renames 6af40ed722
```

Run the CPU metadata test in an environment with `torch`:

```bash
python3 -m pytest test/registered/attention/unittests/mamba/test_mamba2.py::TestMambaTrackSsmMetadata -q
```

Run broader Mamba2 CUDA coverage:

```bash
python3 -m pytest test/registered/attention/unittests/mamba/test_mamba2.py -q
python3 -m pytest test/registered/layers/mamba/test_mamba_ssm_ssd.py -q
```

For PR validation, collect before/after results on the same GPU machine:

- Mamba2/NemotronH `--mamba-scheduler-strategy extra_buffer` accuracy sanity.
- Before/after speed benchmark for the same model, prompts, batch/concurrency,
  and server flags.
- If possible, add a kernel-level exact-state regression for multi-request
  non-flat-chunk-aligned tracking, comparing tracked state with per-request rerun
  truncated to the snapshot length.

## PR Testing Text

```md
## Testing

Non-GPU checks completed:

- `git diff --check`: passed
- `python3 -m compileall -q ...`: passed on changed Python files
- `python3 -m ruff check --select F401,F821,F822,F823,F841 ...`: passed on changed Python files
- `SKIP=rustfmt-sgl-model-gateway,rustfmt-sgl-router PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files`: passed

Full `pre-commit run --all-files` could not complete on this machine because
`rustup` and `cargo` are not installed; only the two Rust formatting hooks were
skipped in the passing rerun.

`pytest test/registered/attention/unittests/mamba/test_mamba2.py::TestMambaTrackSsmMetadata -q`
could not run on this machine because `torch` is not installed.

## Pending GPU Validation

- Mamba2 metadata/unit tests in a `torch` environment
- Mamba2 Triton/kernel exact-state regression for multi-request non-flat-chunk-aligned tracking
- NemotronH/Mamba2 `extra_buffer` accuracy sanity
- before/after speed benchmark
```
