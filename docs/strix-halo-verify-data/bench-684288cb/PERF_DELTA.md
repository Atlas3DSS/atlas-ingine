# Standalone bench: 684288cb (native WMMA) vs 2a1fa4f2 (scalar SIMT) — same box, same model, full-default battery
| metric | 2a1fa4f2 | 684288cb | delta |
|---|---:|---:|---:|
| decode_code_tok_s | 23.03 | 28.36 | +23% |
| decode_prose_tok_s | 23.02 | 28.38 | +23% |
| decode_edit_tok_s | 22.98 | 28.34 | +23% |
| decode_tool_tok_s | 23.46 | 28.50 | +21% |
| decode_multilang_tok_s | 23.00 | 28.37 | +23% |
| agent_ttft_30k_96k_s | 8.96 | 0.45 | -95% |
| agent_turn_30k_96k_s | 13.19 | 4.11 | -69% |
| cold_34k_s | 806.88 | 46.58 | -94% |
| prefill_8k/32k/131k_tok_s (prefix-cache-dominated) | 3.42M/4.79M/5.58M | 3.43M/4.85M/5.51M | ~flat |

Cold server_ready_s unchanged (~10.1 s — load time, unaffected). All eligibility notes satisfied on both runs. Summary prefill tok_s fields remain prefix-cache-dominated (per prior PR note).
