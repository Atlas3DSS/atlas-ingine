"""Standalone CPU unit tests for blender_celebration_hillclimb (Verified Ledger + 3-Tier Compressor + Credits)."""
from __future__ import annotations

import os
import tempfile
import unittest

from PIL import Image

from .context_compressor import IterationDelta, RollingContextCompressor
from .ledger import CelebrationVideoLedger
from .verifier import apply_editorial_titles_and_credits


class TestBlenderCelebrationHillclimb(unittest.TestCase):
    def test_01_verified_ledger_addition_superseding_and_retraction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vl_celeb_") as td:
            led = CelebrationVideoLedger(os.path.join(td, "celebration_video_ledger.sqlite"))
            try:
                rej, _ = led.admit_entry(
                    category="scene_technique",
                    slot_key="lighting_rig",
                    title="Unverified Speculation",
                    summary="Must be rejected without vision verification.",
                    iter_num=1,
                    score=70.0,
                    host_verified=True,
                    vision_verified=False,
                )
                self.assertIsNone(rej)

                id1, sup1 = led.admit_entry(
                    category="scene_technique",
                    slot_key="lighting_rig",
                    title="Lighting Rig v1",
                    summary="Area key + emerald rim.",
                    iter_num=1,
                    score=70.0,
                    host_verified=True,
                    vision_verified=True,
                )
                self.assertIsNotNone(id1)
                self.assertEqual(sup1, [])

                id2, sup2 = led.admit_entry(
                    category="scene_technique",
                    slot_key="lighting_rig",
                    title="Lighting Rig v2",
                    summary="Higher-scoring rig supersedes v1.",
                    iter_num=2,
                    score=84.0,
                    host_verified=True,
                    vision_verified=True,
                )
                self.assertEqual(sup2, [id1])

                retracted = led.retract_entry(entry_id=id2, iter_num=3, reason="Retracted after visual glare defect")
                self.assertEqual(retracted, [id2])
                rendered = led.render_tier1_ledger()
                self.assertIn("RETRACTED #", rendered)
                st = led.stats()
                self.assertEqual(st["events_by_action"].get("SUPERSEDE"), 1)
                self.assertEqual(st["events_by_action"].get("RETRACT"), 1)
            finally:
                led.close()

    def test_02_rolling_context_compressor_bounded_across_24_iters(self) -> None:
        comp = RollingContextCompressor(max_tier2_deltas=8, max_tier2_chars=2600)
        deltas = [
            IterationDelta(
                iter_num=i,
                host_ok=True,
                host_score=46.0,
                vision_score=24.0 + (i % 10),
                composite_score=70.0 + (i % 10),
                delta_vs_best=1.0,
                is_new_best=(i % 5 == 0),
                visual_verdict=f"Iter {i} visual critique.",
            )
            for i in range(1, 25)
        ]
        t2 = comp.build_tier2_history(deltas)
        self.assertLessEqual(len(t2), 2600)
        self.assertIn("Earlier epoch (iters 1..16)", t2)
        self.assertIn("Iter #24", t2)

    def test_03_editorial_titles_and_credits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vl_titles_") as td:
            src, dst = os.path.join(td, "src"), os.path.join(td, "dst")
            os.makedirs(src)
            for i in range(1, 5):
                Image.new("RGB", (640, 360), (16, 21, 24)).save(os.path.join(src, f"frame_{i:04d}.png"))
            n = apply_editorial_titles_and_credits(src_frames_dir=src, dst_frames_dir=dst, fps=8, end_card_seconds=1.0)
            self.assertEqual(n, 12)


if __name__ == "__main__":
    unittest.main()
