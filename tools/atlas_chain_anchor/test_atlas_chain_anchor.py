#!/usr/bin/env python3
"""Unit tests for atlas_chain_anchor.py (Merkle proofs, tamper rejection, SQLite ledger summary, calldata)."""
import json
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parent / "atlas_chain_anchor.py"
sys.path.insert(0, str(SCRIPT.parent))
import atlas_chain_anchor as aca  # noqa: E402


class TestAtlasChainAnchor(unittest.TestCase):
    def test_bundle_verify_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "paper.md").write_text("# Test Paper\n", encoding="utf-8")
            (root / "mesh.glb").write_bytes(b"glTF\x02\x00\x00\x00MESH_DATA")
            db_path = root / "verified_ledger.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE checkpoints (id INTEGER PRIMARY KEY, sha TEXT)")
            conn.execute("INSERT INTO checkpoints VALUES (1, 'abc123')")
            conn.commit()
            conn.close()

            out = root / "bundle.json"
            res = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "bundle",
                    "--tag", "test-v1",
                    "--creator", "atlas3d.eth",
                    "--aliases", "eacc.eth",
                    "--uri", "https://github.com/Atlas3DSS/atlas-ingine/releases/tag/v0.3-omni-grafting-prior-art",
                    "--root-dir", str(root),
                    "--output", str(out),
                    str(root / "paper.md"), str(root / "mesh.glb"), str(db_path),
                ],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("MERKLE_ROOT=0x", res.stdout)
            bundle = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(bundle["leaf_count"], 3)
            sq_leaf = [e for e in bundle["leaves"] if e["record"]["logical_path"] == "verified_ledger.sqlite"][0]
            self.assertEqual(sq_leaf["record"]["ledger_summary"]["sqlite_row_counts"]["checkpoints"], 1)

            # Verify whole bundle + single file membership
            subprocess.run([sys.executable, str(SCRIPT), "verify", "--bundle", str(out)], check=True, capture_output=True)
            subprocess.run(
                [sys.executable, str(SCRIPT), "verify", "--bundle", str(out), "--file", str(root / "mesh.glb")],
                check=True, capture_output=True,
            )

            # Calldata roundtrip
            decoded = aca.decode_self_calldata(bundle["onchain_anchor"]["self_calldata_v1"]["data_hex"])
            self.assertEqual(decoded["merkle_root"], bundle["merkle_root"])
            self.assertEqual(decoded["meta"]["creator"], "atlas3d.eth")

            # Tamper 1 byte in record -> proof must fail
            tampered = dict(sq_leaf["record"])
            tampered["size_bytes"] += 1
            self.assertFalse(aca.verify_merkle_proof(tampered, sq_leaf["proof"], bundle["merkle_root"]))


if __name__ == "__main__":
    unittest.main()
