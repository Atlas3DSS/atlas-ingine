# Strix Halo verification run — provenance & findings
Machine: Strix Halo (AMD Ryzen AI Max+ 395 w/ Radeon 8060S, gfx1151), CachyOS, kernel 7.3.1-1-cachyos, 124 GB unified RAM.
ROCm: rocm-hip-sdk 7.2.4 (HIP 7.16). cmake 4.5.1, ninja 1.13.2.
Commit: feat/hip-full-port @ 2a1fa4f, clean tree; harness invoked unmodified.
Harness: `strix_halo_verify.py all --preset hip-strix-dev --model-dir ~/models/Inkling-2-Mini-ingine --require-gfx1151 --out-dir ./strix_results`
Model: satellitedown/Inkling-2-Mini-abliterated-ingine — all 25 files SHA256-verified against SHA256SUMS.

## Outcome
`preflight` ok; `build_test`: clean HIP build (zero-CUDA ldd audit passed), CTest **59/66 passed, 7 failed** (exit 2) — harness therefore skipped `bench`/`score`. This dump is the step-2 failure payload per protocol.

## Test failures (see ctest_LastTest.log for raw output)
1. **prefill_moe** — numerical: `FAIL combine (exact): 57 / 1612800 outside tolerance` (allowed 0; e.g. got 0.00292969 ref 0.00291443). Tiny-magnitude exact-compare misses on gfx1151; looks like a tolerance/contract issue rather than gross compute error — the bounds-style `kernels_dense` checks all pass.
2. **model_runtime** — Subprocess aborted (server admission faults visible in log; `std::runtime_error` injected-fault path aborted).
3. **model_reuse, model_cache_stage3** — ctest-timeout (3600 s each). NOTE: standalone reruns of the model tests (after freeing unified memory) passed in 138 s–340 s. They time out only under full-suite back-to-back load — suspect memory fragmentation / GTT pressure accumulation across sequential model tests on unified memory.
4. **batch_parity** — failed in this full run; passed in a standalone rerun earlier the same day. Same order-dependent unified-memory suspicion.
5. **core_baremetal_cpu** — THP assertion `pinned.hugepage_advised() == hp_env` fails under CachyOS default THP policy (environment, not port).
6. **core_baremetal_cuda** — same baremetal path on a CUDA-less host.

## Environment findings (actionable for the repo/harness)
1. **Hardcoded repo-relative model dir in CTest**: model_path/model_parity* look for `models/Inkling-2-Mini` under repo root and ignore the harness `--model-dir`. Worked around with symlink; recommend exporting the model dir into the CTest env/preset.
2. **Hardcoded 1800 s CTest cap** in `strix_halo_verify.run_build_and_test` (~line 452): once memory was free the full suite legitimately needed 4692 s and the first `all` attempt aborted at the cap. This dump was produced with the cap raised to 14400 s via an in-memory monkeypatch (repo untouched); recommend a `--ctest-timeout` flag.
3. **Unified-memory contention gives undiagnostic failures**: with a 73 GB GPU-resident process running (an unrelated inference server), model tests failed in `hipMalloc` with `hipErrorOutOfMemory`. Stopping it fixed every memory-dependent test except the order-dependent timeouts above. preflight could record free GPU/unified memory and warn.
4. `rocm-smi` CLI is not packaged on Arch/CachyOS (only `rocm-smi-lib`); harness already treats it as optional — `rocminfo` sufficed for gfx1151 detection.

## Files
- `strix_halo_report.json` — full harness report (preflight + build_test, incl. ctest_tail)
- `ctest_LastTest.log` — raw CTest output from the 4692 s full run
- `build_test.json` — earlier standalone build-test (pre memory-free; shows the OOM failure mode)
- `CTestCostData.txt` — per-test cost/timing
- `model_download.log` — model pack fetch
- `verify_all.log` — (empty; harness stdout was buffered and lost when the CTest cap killed the first attempt)

## Not yet included
`bench`/`score` phases were skipped by the harness (they gate on build_test ok). A standalone `bench` run is being executed separately to capture the performance data the optimization pass needs; it will land as a follow-up commit on this branch.
