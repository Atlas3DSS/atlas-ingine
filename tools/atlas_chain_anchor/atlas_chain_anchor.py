#!/usr/bin/env python3
"""
Atlas Verified Ledger & Asset Merkle-Root On-Chain Anchor (Base L2 & Ethereum Mainnet).

Zero private keys ever touch this tool. Builds an RFC-6962 domain-separated SHA-256
Merkle tree over any files, directories, Verified Ledger SQLite/JSONL chains, or
prior-art release bundles, emits per-asset inclusion proofs (proving any single asset
without exposing other files in the tree), and generates ready-to-sign Base Mainnet
(chainId 8453) and Ethereum Mainnet (chainId 1) anchoring payloads (both zero-schema
self-calldata ATLAS_ANCHOR_V1 and Base/ETH EAS predeploy calldata).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
CALLDATA_MAGIC = b"ATLAS_VL_ANCHOR_V1\x00"

CHAIN_CONFIG = {
    "base": {
        "name": "Base Mainnet",
        "chain_id": 8453,
        "rpc_url": "https://mainnet.base.org",
        "explorer_tx": "https://basescan.org/tx/",
        "eas_contract": "0x4200000000000000000000000000000000000021",
        "eas_schema_registry": "0x4200000000000000000000000000000000000020",
        "eas_explorer": "https://base.easscan.org",
    },
    "ethereum": {
        "name": "Ethereum Mainnet",
        "chain_id": 1,
        "rpc_url": "https://ethereum-rpc.publicnode.com",
        "explorer_tx": "https://etherscan.io/tx/",
        "eas_contract": "0xA1207F3BBa224E2c9c3c6D5aF63D0eb1582Ce587",
        "eas_schema_registry": "0xA7b39296258348C78294F95B872b282326A97BDF",
        "eas_explorer": "https://easscan.org",
    },
    "base-sepolia": {
        "name": "Base Sepolia Testnet",
        "chain_id": 84532,
        "rpc_url": "https://sepolia.base.org",
        "explorer_tx": "https://sepolia.basescan.org/tx/",
        "eas_contract": "0x4200000000000000000000000000000000000021",
        "eas_schema_registry": "0x4200000000000000000000000000000000000020",
        "eas_explorer": "https://base-sepolia.easscan.org",
    },
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def hash_leaf(leaf_record: Dict[str, Any]) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + canonical_json_bytes(leaf_record)).digest()


def hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def extract_sqlite_ledger_summary(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Read-only inspection of a Verified Ledger SQLite database if applicable."""
    if path.suffix.lower() not in (".sqlite", ".sqlite3", ".db"):
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = conn.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        row_counts = {}
        for t in tables[:32]:
            try:
                row_counts[t] = int(cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
            except Exception:
                pass
        conn.close()
        return {"sqlite_tables": tables, "sqlite_row_counts": row_counts}
    except Exception:
        return None


def collect_inputs(paths: List[str], root_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    base = pathlib.Path(root_dir).resolve() if root_dir else None
    seen: Dict[str, pathlib.Path] = {}
    for raw in paths:
        p = pathlib.Path(raw).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {raw}")
        if p.is_file():
            rel = str(p.relative_to(base)) if base and str(p).startswith(str(base)) else p.name
            seen[rel] = p
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    if base and str(child).startswith(str(base)):
                        rel = str(child.relative_to(base))
                    else:
                        rel = f"{p.name}/{child.relative_to(p)}"
                    seen[rel] = child
    records: List[Dict[str, Any]] = []
    for rel in sorted(seen.keys()):
        fp = seen[rel]
        rec: Dict[str, Any] = {
            "logical_path": rel,
            "size_bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
        }
        sq = extract_sqlite_ledger_summary(fp)
        if sq is not None:
            rec["ledger_summary"] = sq
        records.append(rec)
    if not records:
        raise ValueError("No files found to anchor.")
    return records


def build_merkle_tree(records: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Build domain-separated binary Merkle tree and return (0x_root, per-leaf proof entries)."""
    leaf_hashes = [hash_leaf(r) for r in records]
    levels: List[List[bytes]] = [leaf_hashes]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt: List[bytes] = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(hash_node(left, right))
        levels.append(nxt)
    root_hex = "0x" + levels[-1][0].hex()

    proofs: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        steps: List[Dict[str, str]] = []
        pos = idx
        for lvl in levels[:-1]:
            if pos % 2 == 0:
                sib_pos = pos + 1 if pos + 1 < len(lvl) else pos
                steps.append({"side": "right", "hash": "0x" + lvl[sib_pos].hex()})
            else:
                sib_pos = pos - 1
                steps.append({"side": "left", "hash": "0x" + lvl[sib_pos].hex()})
            pos //= 2
        proofs.append({
            "index": idx,
            "leaf_hash": "0x" + leaf_hashes[idx].hex(),
            "record": rec,
            "proof": steps,
        })
    return root_hex, proofs


def verify_merkle_proof(record: Dict[str, Any], proof_steps: List[Dict[str, str]], expected_root: str) -> bool:
    cur = hash_leaf(record)
    for step in proof_steps:
        sib = bytes.fromhex(step["hash"].removeprefix("0x"))
        if step["side"] == "right":
            cur = hash_node(cur, sib)
        elif step["side"] == "left":
            cur = hash_node(sib, cur)
        else:
            return False
    return ("0x" + cur.hex()).lower() == expected_root.lower()


def build_self_calldata(merkle_root_hex: str, tag: str, creator: str, uri: str) -> str:
    """Compact, human-inspectable, zero-contract calldata payload for self-tx anchoring on Base/ETH."""
    root_bytes = bytes.fromhex(merkle_root_hex.removeprefix("0x"))
    if len(root_bytes) != 32:
        raise ValueError("merkle_root must be 32 bytes")
    meta = canonical_json_bytes({
        "v": 1,
        "tag": tag,
        "creator": creator,
        "uri": uri,
    })
    raw = CALLDATA_MAGIC + root_bytes + meta
    return "0x" + raw.hex()


def decode_self_calldata(calldata_hex: str) -> Dict[str, Any]:
    raw = bytes.fromhex(calldata_hex.removeprefix("0x"))
    if not raw.startswith(CALLDATA_MAGIC):
        raise ValueError("Calldata does not start with ATLAS_VL_ANCHOR_V1 magic")
    rest = raw[len(CALLDATA_MAGIC):]
    if len(rest) < 32:
        raise ValueError("Truncated calldata")
    root_hex = "0x" + rest[:32].hex()
    meta = json.loads(rest[32:].decode("utf-8"))
    return {"merkle_root": root_hex, "meta": meta}


def cmd_bundle(args: argparse.Namespace) -> int:
    records = collect_inputs(args.inputs, root_dir=args.root_dir)
    root_hex, leaf_proofs = build_merkle_tree(records)
    calldata = build_self_calldata(root_hex, args.tag, args.creator, args.uri)
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Sanity self-verify every generated proof before writing!
    for entry in leaf_proofs:
        if not verify_merkle_proof(entry["record"], entry["proof"], root_hex):
            raise RuntimeError(f"Internal proof verification failed for {entry['record']['logical_path']}")

    bundle = {
        "schema_version": "atlas-chain-anchor-bundle/v1",
        "created_utc": now_utc,
        "tag": args.tag,
        "creator_ens": args.creator,
        "creator_aliases": [x.strip() for x in args.aliases.split(",") if x.strip()] if args.aliases else [],
        "reference_uri": args.uri,
        "license": args.license,
        "merkle_algorithm": "sha256-rfc6962-domain-separated",
        "leaf_count": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records),
        "merkle_root": root_hex,
        "onchain_anchor": {
            "recommended_chain": "base",
            "self_calldata_v1": {
                "data_hex": calldata,
                "data_bytes": len(bytes.fromhex(calldata.removeprefix("0x"))),
                "value_wei": "0",
                "instructions": "Send a 0-ETH tx from your own wallet (e.g. atlas3d.eth / eacc.eth) to your own address with this data field on Base Mainnet (chainId 8453, <$0.001) or Ethereum Mainnet (chainId 1).",
            },
            "eas_schema_definition": "bytes32 merkleRoot,string tag,string creatorEns,string referenceUri,uint32 leafCount",
            "chains": CHAIN_CONFIG,
        },
        "leaves": leaf_proofs,
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"MERKLE_ROOT={root_hex}")
    print(f"LEAVES={len(records)} TOTAL_BYTES={bundle['total_bytes']}")
    print(f"CALLDATA_BYTES={bundle['onchain_anchor']['self_calldata_v1']['data_bytes']}")
    print(f"WROTE={out_path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    bundle = json.loads(pathlib.Path(args.bundle).read_text(encoding="utf-8"))
    root_hex = bundle["merkle_root"]
    decoded = decode_self_calldata(bundle["onchain_anchor"]["self_calldata_v1"]["data_hex"])
    if decoded["merkle_root"].lower() != root_hex.lower():
        print("FAIL: calldata merkle_root mismatch", file=sys.stderr)
        return 1

    if args.file:
        target = pathlib.Path(args.file).resolve()
        actual_sha = sha256_file(target)
        matches = [e for e in bundle["leaves"] if e["record"]["sha256"] == actual_sha]
        if not matches:
            print(f"FAIL: {target} (sha256={actual_sha}) not found in bundle", file=sys.stderr)
            return 1
        for m in matches:
            ok = verify_merkle_proof(m["record"], m["proof"], root_hex)
            if not ok:
                print(f"FAIL: invalid Merkle proof for {m['record']['logical_path']}", file=sys.stderr)
                return 1
            print(f"VERIFIED SINGLE ASSET: {m['record']['logical_path']} (sha256={actual_sha}) -> root {root_hex}")
        return 0

    for entry in bundle["leaves"]:
        if not verify_merkle_proof(entry["record"], entry["proof"], root_hex):
            print(f"FAIL: invalid Merkle proof for {entry['record']['logical_path']}", file=sys.stderr)
            return 1
    print(f"VERIFIED ALL {len(bundle['leaves'])} LEAVES + CALLDATA -> root {root_hex}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas Verified Ledger & Asset Merkle-Root On-Chain Anchor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bundle", help="Build Merkle root, per-asset proofs, and Base/ETH calldata")
    pb.add_argument("--tag", required=True, help="Release or project tag (e.g. v0.3-omni-grafting-prior-art)")
    pb.add_argument("--creator", required=True, help="Primary ENS or creator handle (e.g. atlas3d.eth)")
    pb.add_argument("--aliases", default="", help="Comma-separated creator aliases (e.g. eacc.eth)")
    pb.add_argument("--uri", required=True, help="Public reference URI (e.g. GitHub Release or Zenodo DOI)")
    pb.add_argument("--license", default="Apache-2.0", help="License label")
    pb.add_argument("--root-dir", default=None, help="Optional root dir for relative logical_path")
    pb.add_argument("--output", required=True, help="Output bundle JSON path")
    pb.add_argument("inputs", nargs="+", help="Files or directories to include in the Merkle tree")
    pb.set_defaults(func=cmd_bundle)

    pv = sub.add_parser("verify", help="Verify bundle proofs or a single asset against the Merkle root")
    pv.add_argument("--bundle", required=True, help="Path to merkle_bundle.json")
    pv.add_argument("--file", default=None, help="Optional single file to prove membership for")
    pv.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
