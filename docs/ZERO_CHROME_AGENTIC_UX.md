# Zero-Chrome Agentic UX — Alpha Suite

> **Status: Alpha (pre-release).** The creative applications documented on this page — **Blender by Atlas (Alpha)**, **Godot by Atlas (Alpha)**, **DragonFruit by Atlas (Alpha)**, and the **Neuron Lens × Verified Ledger UX (Alpha)** — are active Alpha builds wired to local `atlas-ingine` (`ingine-serve`) + Verified Ledger backends. Upstream forks preserve their original licenses (`GPL-3.0`, `MIT`, `AGPL-3.0`) and communicate strictly across process boundaries over authenticated loopback HTTP IPC.

## Design Contract

Traditional 3D creation, game-engine, and resin-slicing suites open onto dense toolbars, property inspectors, and modal dialogs. **Zero-Chrome Agentic UX** inverts the default layout across every Atlas-adapted creative tool while keeping 100% of the native application accessible on demand:

1. **Full-bleed viewport + single Omni bar at rest (`Creative Mode`).** Every application opens directly into a 100% viewport with a single floating prompt/voice/reference bar (`say it, type it, sketch it, or drop a reference`) anchored at the bottom. Zero docks, toolbars, welcome modals, or property panels are drawn outside the Omni bar mask.
2. **Instant reversible `Power User Mode` (`Ctrl+F10`).** Pressing `Ctrl+F10` (or clicking `just create Ctrl+F10` / `ask omni`) toggles the full native interface without reloading or mutating scene state. Bare `F10` is ignored so native application shortcuts remain intact; corner-reveal gestures and `L` temporarily surface the health and Verified Ledger strips and retract cleanly.
3. **Automated pixel-diff `zero_chrome_test` gate.** Every Alpha build is verified against a headless no-chrome reference render across three states (`at_rest`, `after_reveal_cycle`, and `after_power_user_roundtrip`), asserting **`0` differing pixels** outside the bounded Omni bar mask.
4. **Host-enforced Verified Ledger + cross-app SHA-256 bridges.** Agent-proposed edits, sculpt strokes, and cross-app asset handoffs commit monotonically through a host-verified ledger (`chk-*` checkpoints, SHA-256 payload pinning, automatic rollback on degraded retries).

---

## Alpha Suite Overview

All three desktop Alpha screenshots below track the **same hero asset (`Atlas Orrery`, 10,832 watertight manifold triangles)** moving live from **Blender by Atlas (Alpha)** to **Godot by Atlas (Alpha)** via `.glb` bridge and to **DragonFruit by Atlas (Alpha)** via `.stl` + protected hero-mask bridge.

| Application | Status | Upstream Base & License | `zero_chrome_test` (`at_rest` / `after_reveal` / `roundtrip`) | Verified Bridge & Ledger Evidence |
|---|---|---|---|---|
| **Blender by Atlas (`Alpha`)** | Pre-release Alpha | Blender `5.2.5 LTS` (`GPL-3.0`) | `0` / `0` / `0` out-of-mask diff pixels (`1440×900`, `1` creative area ↔ `4` power-user areas) | `18/18` watertight manifold Omni-Brush matrix (`88/88` Verified Ledger checkpoints); closed-loop out-of-process MIT TRELLIS.2-4B worker |
| **Godot by Atlas (`Alpha`)** | Private preview Alpha (`Atlas3DSS/godot-by-atlas`) | Godot `4.7.2` (`MIT`) | `0` / `0` / `0` out-of-mask diff pixels (`1440×900`, `24 px` mask margin) | Live loopback `.glb` push (`10,832/10,832` tris, `12` meshes, `6` materials, SHA-256 verified) + 31-bone 4-clip rigged `.glb` import parity |
| **DragonFruit by Atlas (`Alpha`)** | Pre-release Alpha | Open-Resin-Alliance/DragonFruit (`AGPL-3.0`) | `0` / `0` / `0` diff pixels (`>0` threshold) across `1,190,160` outside-Omni pixels (`1440×900`) | Live `.stl` + `3,372`-face protected hero-mask handoff (`0` hero heavy-contact violations); **out-of-process IPC boundary** to closed-source commercial `Atlas Support Engine` |
| **Neuron Lens × Verified Ledger (`Alpha`)** | Public Alpha in this repo | Standalone HTML + `ingine-serve` (`Apache-2.0`) | Single-surface live chat + telemetry drawer | [`zero_chrome_neuron_lens_ux.html`](../modules/zero_gradient_omni_modal_grafting/adversarial_ledger_proof/zero_chrome_neuron_lens_ux.html) — `0/40` committed adversarial breaches vs `22/40` prompt-only |

---

## 1. Blender by Atlas (`Alpha` — Blender `5.2.5 LTS`, `GPL-3.0`)

| Creative Mode (Zero-Chrome Default) | Power User Mode (`Ctrl+F10` Instant Toggle) |
|---|---|
| ![Blender by Atlas Alpha — Creative Mode](media/zero_chrome_ux/blender_alpha_creative_mode.png) | ![Blender by Atlas Alpha — Power User Mode](media/zero_chrome_ux/blender_alpha_power_user_mode.png) |
| Single full-bleed viewport (`1` area) with the floating Omni bar (`say it, type it, sketch it, or drop a reference`). `0` differing pixels outside the Omni mask across rest, reveal, and `Ctrl+F10` round-trip. | Full native Blender `5.2.5 LTS` layout (`4` areas, Scene Collection `AtlasHero`, Transform inspector, Timeline) with the `just create` pill in the top header. |

### Omni-Brush Live-Stroke Matrix (`18/18` Watertight Manifold, `88/88` Verified Ledger Checkpoints)

Inside Blender by Atlas (`Alpha`), natural-language prompts compile to live multi-point sculpt strokes, Meshmixer-class surgery operators, and 16-bit alpha stamps — each guarded by a hard physical wall-thickness floor (`>= 1.2 mm`), watertight manifold checks, and automatic Verified Ledger rollback:

![Blender by Atlas 5.2.5 LTS — 18-Brush Omni-Brush Live-Stroke Matrix](media/zero_chrome_ux/blender_alpha_omni_brush_18_matrix.png)

- **9 multi-point sculpt brushes (`>= 1.2 mm` wall guard):** Clay/Draw Build, Crease, Pinch, Inflate, Flatten/Scrape, Smooth, Grab/Drag, Mask + Thicken, Symmetry-X (`12.0 mm` inward carve on a `4.0 mm` plate clamps cleanly at `1.20 mm`).
- **6 transactional Meshmixer surgery brushes:** Hollow + Drain (`72.7%` resin saved, `2.0 mm` wall, `0` trapped pockets), Keyed Cut & Peg (`6.0 mm` peg / `6.44 mm` socket, `0.22 mm` clearance), Support-Scar Heal (`-68.4%` roughness), Inspector Auto-Repair (`-2` floating islands, `8` sealed boundary edges), Kitbash Stamp-Weld, and Masked TRELLIS Weld (calling an out-of-process MIT TRELLIS.2-4B worker over authenticated loopback HTTP — zero GPLv3/MIT mixing).
- **3 16-bit alpha brushes:** Prompt-to-Alpha Stamp, Seamless Alpha Rake (`0.0` seam), and Photo-to-Alpha (De-Lit `-76.0%`).

---

## 2. Godot by Atlas (`Alpha` — Godot `4.7.2`, `MIT`)

| Creative Mode (Zero-Chrome Default) | Power User Mode (`Ctrl+F10` Instant Toggle) |
|---|---|
| ![Godot by Atlas Alpha — Creative Mode](media/zero_chrome_ux/godot_alpha_creative_mode.png) | ![Godot by Atlas Alpha — Power User Mode](media/zero_chrome_ux/godot_alpha_power_user_mode.png) |
| Full-bleed `1440×900` Godot 3D viewport showing the `Atlas Orrery` received live from Blender by Atlas, with `0` out-of-mask diff pixels across rest, reveal, and round-trip. | Native Godot `4.7.2.stable.atlas` editor (`orrery_handoff_showcase.tscn`) with Scene tree (`AtlasOrreryHero` + 12 meshes), FileSystem, Inspector, bottom `VERIFIED LEDGER \| 01 import atlas_orrery_hero` bar, and `ask omni` / `just create Ctrl+F10` top-right controls. |

- **Authenticated Blender → Godot live `.glb` bridge:** Loopback HTTP push (`POST /v1/bridge/import`, SHA-256 `a800038f11df220b...`) imports `Atlas Orrery` with exact geometry parity (`10,832/10,832` triangles, `5,664` vertices, `12` `MeshInstance3D` nodes, `6` unique PBR materials, Y-up bounds `[0.11514, 0.14943, 0.11514] m`, Verified Ledger seq `1`).
- **Rigged character bridge verification:** Tested end-to-end on a 31-bone A-pose character (`22` humanoid + `6` wing + `3` tail bones, `4` animation clips `Atlas_idle`/`Atlas_walk`/`Atlas_slash`/`Atlas_wave`) with byte-identical SHA-256 transfer (`618b816f1bf72073...`) and intact `Skeleton3D` / `AnimationPlayer` tracks.
- **Repository status:** `Atlas3DSS/godot-by-atlas` is currently a **private Alpha preview** repository while Alpha integration stabilizes.

---

## 3. DragonFruit by Atlas (`Alpha` — Open-Resin-Alliance/DragonFruit, `AGPL-3.0`)

| Creative Mode (Zero-Chrome Default) | Power User Mode (`Ctrl+F10` Instant Toggle) |
|---|---|
| ![DragonFruit by Atlas Alpha — Creative Mode](media/zero_chrome_ux/dragonfruit_alpha_creative_mode.png) | ![DragonFruit by Atlas Alpha — Power User Mode](media/zero_chrome_ux/dragonfruit_alpha_power_user_mode.png) |
| Full-bleed `1440×900` resin build plate viewport displaying `Atlas Orrery Miniature` (`10,832` triangles) with only the floating Omni bar rendered (`0/1,190,160` diff pixels outside mask across rest, corner-linger, and `Ctrl+F10` round-trip). | Native DragonFruit slicer UI (`Prepare / Support / Export / Printing`, `Atlas Studio Plate`, `1,723` layers, bottom `plate / islands / suction / peel` health strip, `Verified Ledger` bar, `ask omni`, and `just create Ctrl+F10`). |

- **Verified Zero-Chrome contract (`atlas-zero-chrome-test/v1.1`):** `opening_project_loaded=true`, `power_user_welcome_card_absent=true`, `bare_f10_ignored=true`, `corner_reveal_passed=true`, `ctrl_f10_roundtrip_passed=true`, `l_reveal_passed=true`, `max_channel_delta_outside_omni=0`, `page_errors=[]`.
- **Blender → DragonFruit protected hero-mask bridge:** Loopback HTTP handoff (`atlas-mesh-handoff/v1`) carries the watertight `.stl` (`10,832` triangles) plus SHA-256-pinned hero-face mask (`3,372` protected faces across `AtlasHeroCrystal`, `AtlasLimeCore`, `AtlasMiddleBrassRing`, `AtlasPedestalBrassCollar`) enforcing `0` heavy-contact support violations on protected surfaces.
- **Strict License & Commercial IP Boundary:** `DragonFruit by Atlas (Alpha)` is strictly an open-source `AGPL-3.0` UI/slicer shell. It communicates **solely via documented out-of-process localhost HTTP IPC** with external services (`ingine-serve`, external CLI `UVtoolsCmd`, and the separate **closed-source commercial `Atlas Support Engine`**). No proprietary `Atlas Support Engine` source code, orientation neural weights, artist support profiles, or `.lys` export internals are included in any public repository.

---

## 4. Neuron Lens × Verified Ledger Zero-Chrome UX (`Alpha` — Public in This Repo)

The same single-surface Zero-Chrome pattern ships directly in this repository for inspecting live `ingine-serve --persona-probe-mode` MoE routing telemetry alongside Host-Enforced Verified Ledger receipts:

![Neuron Lens x Verified Ledger Zero-Chrome UX Alpha Preview](../modules/zero_gradient_omni_modal_grafting/adversarial_ledger_proof/live_results_v2/zero_chrome_neuron_lens_ux_preview.png)

- **Open locally:** [`modules/zero_gradient_omni_modal_grafting/adversarial_ledger_proof/zero_chrome_neuron_lens_ux.html`](../modules/zero_gradient_omni_modal_grafting/adversarial_ledger_proof/zero_chrome_neuron_lens_ux.html) (loads the verified `v2` live trace offline and connects live to `http://127.0.0.1:18099/v1/probe/chat` when `ingine-serve --persona-probe-mode` is running).
- **Measured result:** Under 40 live Pliny-style adversarial prompts (`20` single-turn + `20` hard `L1B3RT4S` many-shot + `10` benign controls), a prompt-only constitution breached **`22/40` (`55.0%`)** while the Host-Enforced Verified Ledger committed **`0/40` (`0.0%`)** breaches at **`10/10` (`100.0%`)** benign utility. Full proof and reproduction commands: [`HOST_ENFORCED_ALIGNMENT.md`](../modules/zero_gradient_omni_modal_grafting/HOST_ENFORCED_ALIGNMENT.md).

---

## Media Checksum Manifest

All 7 Alpha UI PNGs in [`docs/media/zero_chrome_ux/`](media/zero_chrome_ux/) are byte-identical copies of their verified test outputs, pinned in [`docs/media/zero_chrome_ux/MANIFEST.sha256.json`](media/zero_chrome_ux/MANIFEST.sha256.json).
