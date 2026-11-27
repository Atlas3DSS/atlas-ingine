# Retest of feat/hip-full-port @ 684288cb (gfx1151)
Same Strix Halo box / model / harness command as commit 1, using the designer's retest command.
Result: CTest **56/66 passed, 5 failed** (run 1 was 59/66 / 7 failed). Wall time 3136 s (down from 4692 s on run 1 — WMMA hot kernels faster). Harness now propagates --model-dir and auto-sizes --max-ctx; --ctest-timeout 7200 used. `all` again gates bench/score on build_test.ok, so bench was skipped here; a standalone bench of 684288cb (the WMMA before/after perf the change targets) follows as its own commit.

## Verdict: 5 of 7 original failures FIXED, but 3 new regressions appeared (all consistent with the new native-WMMA work)
Fixed vs commit 1:
- batch_parity (B=64) — PASS
- core_baremetal_cuda — PASS
- model_runtime (capture_switch abort) — PASS
- model_cache_stage3 — PASS (was a timeout)
- prefill_moe (combine 1-ULP) — PASS

## Still failing (2)
1. core_baremetal_cpu — NEW line 219 now (`dbuf.hugepage_advised() == hp_env`); the hugepage mmap port works (AnonHugePages=4.0/4.0 MiB on PinnedHostBuffer AND MappedHostBuffer) but the assertion still trips under CachyOS default THP policy. Pure env issue, unchanged root cause.
2. model_reuse — no longer a timeout (completes in 454 s). Now a real diff: `checkpoint restore equals the strict-prefix path with spec (sampled, G 200 / G 1300 tokens)` FAIL; all greedy variants PASS, all parking/interleave cases PASS. Sampled + spec checkpoint-restore only.

## New regressions (3) — likely from native WMMA
3. kernels_moe — NEW. All moe_up pass across nvfp4/mxfp8/atlas2/atlas3/exl3_2; dequant passes. Fails on `moe_down exl3_2 M=1 expert sum` (worst err/tol=0.382, off-rounding=0.1975) and `moe_down exl3_3 M=1 expert sum` (worst err/tol=0.545, off-rounding=0.1532). Note out-of-tol=0 but off-rounding exceeds the exactness bound — the M=1 (decode) moe_down WMMA path for EXL3_2/EXL3_3 diverges.
4. model_parity_generate — NEW + deterministic (greedy, so not sampling noise): `orig_kv16_chat generate: 5/33 greedy tokens match` (top1 0.667 < min 1.0); the abliterated cases are 33/33. So the ORIGINAL model's chat case diverges from golden after 5 tokens on the new path — highest-signal regression to look at.
5. spec_conditional_fallback — NEW: eager path (correctly qualified, SWITCH unsupported on HIP) — 4 sub-checks pass, but `unhooked decode after hooked request restores baseline on eager path` FAILs. Looks like persona-hook state not fully reset on the eager path.

## Files
- strix_halo_report.json — preflight + build_test (incl ctest_tail)
- ctest_LastTest.log — full per-test output for the 684288cb run
