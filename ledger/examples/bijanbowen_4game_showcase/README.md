# Inkling-2-Mini 3-GPU Multi-Agent Array + Shared Verified Ledger — `@bijanbowen` 4-Game Showcase

All four interactive 3D WebGL2 games in this bundle were **100% authored and self-repaired by local Inkling-2-Mini** (zero human-written target game code) served via `atlas-ingine` (`ingine-serve --spec` + native vision tower across 3 local NVIDIA Tesla V100 GPUs at `$0.00` cloud spend).

Each game ran an honest **Stock One-Shot Attempt 0** (bare prompt, no ledger, no vision) followed by a **3-Archetype Parallel Tournament Array** (*Graphics & WebGL2 Shader Architect*, *3D Physics & Gameplay Engineer*, *Visual Polish & VFX Director*) sharing a single 4-bank SQLite **Verified Ledger** (`atlas-ingine/ledger/verified_ledger.py`), native Inkling-2-Mini visual self-review on rendered 3-frame action strips, and a headless Chromium WebGL2 state-machine inspector & live keyboard playtest oracle.

## Results (Stock Attempt-0 vs 3-GPU Array + Shared Verified Ledger)

| Target | Stock Attempt-0 | Array + Verified Ledger Champion | Verified Live Playtest & WebGL2 Telemetry | Interactive Play & Clip |
|---|---|---|---|---|
| **Isometric 4x4 Off-Road Driving Game on a Live 5-Day Stock Price Curve 3D Terrain** (vs Claude) | FAIL `2.0/100` (`TOKEN_TRUNCATION`) | **PASS `95.54/100`** (`round_02/agent_physics`) | Stock 4x4 Off-Road PASS: dist_delta=51.54, prices=[121.07, 142.88, 166.09], max_pitch=0.2917, motion_mad=0.119, colors=479, entropy=4.01 | vision_qa=86/100 | [01_stock_offroad_4x4/final_verified.html](01_stock_offroad_4x4/final_verified.html) · [stock_offroad_4x4_gameplay.mp4](01_stock_offroad_4x4/stock_offroad_4x4_gameplay.mp4) |
| **3D Haunted Bowling Alley Horror Game with Rigid-Body 10-Pin Physics** (vs GLM-5.4) | FAIL `2.0/100` (`TOKEN_TRUNCATION`) | **PASS `84.16/100`** (`round_04/agent_polish`) | Bowling Horror 3D PASS: ball_dz=35.61, pins_knocked=2/10, score=0, motion_mad=0.010, colors=336, entropy=2.77 | vision_qa=82/100 | [02_bowling_horror_3d/final_verified.html](02_bowling_horror_3d/final_verified.html) · [bowling_horror_3d_gameplay.mp4](02_bowling_horror_3d/bowling_horror_3d_gameplay.mp4) |
| **3D Skate-to-GTA Open-City Sandbox (Tricks + Enter/Drive/Exit Cars)** (vs GPT 6.2 Astra) | FAIL `2.0/100` (`TOKEN_TRUNCATION`) | **PASS `92.23/100`** (`round_03/agent_graphics`) | Skate-to-GTA 3D PASS: skate_dist=23.64, tricks=1, score=250, entered_car=True, car_dist=23.67, exited_to_skate=True, colors=305, entropy=3.37 | vision_qa=78/100 | [03_skate_to_gta_city/final_verified.html](03_skate_to_gta_city/final_verified.html) · [skate_to_gta_city_gameplay.mp4](03_skate_to_gta_city/skate_to_gta_city_gameplay.mp4) |
| **High-Speed Multiplayer 3D Arcade Racer (2P Split-Screen + BroadcastChannel + AI Pack)** (vs Gemini Carbon) | FAIL `2.0/100` (`TOKEN_TRUNCATION`) | **PASS `100.0/100`** (`round_02/agent_polish`) | Multiplayer 3D Racing PASS: splitScreen=True, BroadcastChannel=True, P1_dist=54.43, P2_dist=60.98, AI=3, motion_mad=0.098, colors=647, entropy=5.31 | vision_qa=88/100 | [04_multiplayer_3d_racing/final_verified.html](04_multiplayer_3d_racing/final_verified.html) · [multiplayer_3d_racing_gameplay.mp4](04_multiplayer_3d_racing/multiplayer_3d_racing_gameplay.mp4) |

## Shared Verified Ledger Summary
- **Active Host Facts (Bank 1):** 5
- **Admitted Verified Winners (Bank 2):** 4
- **Auto-Linked Error → Repair Rules (Bank 3):** 4
- **Total Logged Failures:** 31
- Interactive Gallery: [index.html](index.html) · SQLite Ledger Export: [verified_ledger_dump.json](verified_ledger_dump.json)
