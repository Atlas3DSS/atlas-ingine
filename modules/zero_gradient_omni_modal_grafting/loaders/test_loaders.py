"""Standalone CPU self-test for drop-in Plato's Key Diffusers & ComfyUI loaders."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from platos_key_schema import (
    INKLING_FORBIDDEN_COORD,
    extract_tap_matrix,
    pack_voltron_cond_packet,
    unpack_and_verify_voltron_cond_packet,
    verify_donor_d2027_zero,
)
from target_registry import TARGET_REGISTRY, get_target_spec

HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class TestPlatosKeyLoaders(unittest.TestCase):
    def test_registry_has_core_targets(self) -> None:
        for key in ("z_image_turbo", "flux2_dev", "flux2_klein_4b", "qwen_image_21"):
            spec = get_target_spec(key)
            self.assertGreater(spec.dit_conditioning_dim, 0)
        self.assertGreaterEqual(len(TARGET_REGISTRY), 6)

    def test_forbidden_sink_coord_guard(self) -> None:
        x = np.zeros((4, 4, 2304), dtype=np.float32)
        verify_donor_d2027_zero(x)
        x_bad = x.copy()
        x_bad[0, 0, INKLING_FORBIDDEN_COORD] = 1.0
        with self.assertRaises(Exception):
            verify_donor_d2027_zero(x_bad)

    def test_extract_tap_matrix_concat_all4(self) -> None:
        x = np.ones((5, 4, 2304), dtype=np.float32)
        x[:, :, INKLING_FORBIDDEN_COORD] = 0.0
        mat = extract_tap_matrix(x, "concat_all4", check_d2027=True)
        self.assertEqual(mat.shape, (5, 9216))

    def test_voltron_vpke_packet_roundtrip(self) -> None:
        cond = np.linspace(-1.0, 1.0, 8 * 64, dtype=np.float32).reshape(8, 64)
        pkt, header = pack_voltron_cond_packet(
            cond,
            target_id="z_image_turbo",
            key_version_sha256="a" * 64,
            donor_tap_spec="concat_all4",
        )
        unpacked, meta = unpack_and_verify_voltron_cond_packet(pkt)
        self.assertEqual(unpacked.shape, (8, 64))
        self.assertEqual(meta["target_id"], "z_image_turbo")
        self.assertGreater(len(pkt), 0)
        self.assertIsInstance(header, dict)
        self.assertLess(float(np.max(np.abs(unpacked - cond.astype(np.float16).astype(np.float32)))), 1e-3)

    @unittest.skipUnless(HAS_SAFETENSORS, "safetensors not installed")
    def test_safetensors_and_loader_imports(self) -> None:
        from platos_key_schema import load_platos_key_v2, save_platos_key_v2
        import comfyui_loader
        import diffusers_loader

        self.assertTrue(callable(save_platos_key_v2))
        self.assertTrue(callable(load_platos_key_v2))
        self.assertIsNotNone(diffusers_loader)
        self.assertIsNotNone(comfyui_loader)


if __name__ == "__main__":
    unittest.main()
