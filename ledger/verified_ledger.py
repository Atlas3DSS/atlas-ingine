#!/usr/bin/env python3
"""Standalone, zero-dependency Host-Verified Continual Memory Ledger.

Works with any OpenAI-compatible inference server (atlas-ingine, atlas-engine,
vLLM, llama.cpp, etc.) and any domain verifier (SQLite, pytest, nvcc/PyTorch,
Three.js/Playwright, Blender, linters, compilers).

Core invariant: the host verifier is the sole write gate.
  1. Bank `facts`   : host-inspected schema, API, or hardware constraints.
  2. Bank `winners` : solutions admitted ONLY after `VerifyResult.ok == True`.
  3. Bank `repairs` : host-observed `error_signature -> rule / fix` pairs.
  4. Bank `drift`   : explicit `env_version` invalidation (marks prior facts &
                      winners stale on schema/API upgrades instead of deleting).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{1,}")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 3}


def _overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    cand = _tokenize(text)
    if not cand:
        return 0.0
    inter = len(query_tokens & cand)
    return inter / (len(query_tokens) ** 0.5 * len(cand) ** 0.5)


def _compact_block(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    used = 0
    for ln in lines:
        add = len(ln) + (1 if kept else 0)
        if used + add > max_chars - 16:
            break
        kept.append(ln)
        used += add
    if kept:
        return "\n".join(kept) + "\n# ... (trimmed)"
    return text[: max(0, max_chars - 3)] + "..."


@dataclass
class VerifyResult:
    """Structured outcome returned by a user-supplied host verifier."""

    ok: bool
    score: float = 0.0
    error_signature: str = ""
    error_detail: str = ""
    summary: str = ""
    facts: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class LoopSummary:
    task_id: str
    solved: bool
    best_score: float
    best_solution: str
    iters_used: int
    records: list[dict[str, Any]] = field(default_factory=list)


class VerifiedLedger:
    """SQLite-backed 4-bank host-verified ledger with deterministic retrieval."""

    def __init__(
        self,
        db_path: str | Path,
        env_version: str = "v1",
        max_context_chars: int = 4000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_context_chars = max(512, int(max_context_chars))
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        cur_ver = self.get_env_version()
        if cur_ver is None:
            self._set_meta("env_version", env_version)
        elif cur_ver != env_version:
            self.bump_env_version(env_version, f"Environment version changed {cur_ver} -> {env_version}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VerifiedLedger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                fact_key    TEXT PRIMARY KEY,
                category    TEXT NOT NULL DEFAULT 'schema',
                value       TEXT NOT NULL,
                env_version TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                updated_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS winners (
                task_id     TEXT PRIMARY KEY,
                task_text   TEXT NOT NULL,
                solution    TEXT NOT NULL,
                score       REAL NOT NULL,
                summary     TEXT NOT NULL DEFAULT '',
                tags_json   TEXT NOT NULL DEFAULT '[]',
                env_version TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                updated_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS repairs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                error_signature TEXT NOT NULL UNIQUE,
                rule            TEXT NOT NULL,
                fix_example     TEXT NOT NULL DEFAULT '',
                hits            INTEGER NOT NULL DEFAULT 1,
                env_version     TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'active',
                updated_at      REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failure_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         TEXT NOT NULL,
                error_signature TEXT NOT NULL,
                error_detail    TEXT NOT NULL,
                env_version     TEXT NOT NULL,
                created_at      REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS drift_notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                old_version TEXT NOT NULL,
                new_version TEXT NOT NULL,
                note        TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    def _set_meta(self, k: str, v: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))
        self.conn.commit()

    def get_env_version(self) -> Optional[str]:
        row = self.conn.execute("SELECT v FROM meta WHERE k = 'env_version'").fetchone()
        return str(row["v"]) if row else None

    # ---- Bank 4: Drift invalidation -----------------------------------------

    def bump_env_version(self, new_version: str, migration_note: str = "") -> dict[str, int]:
        """Mark prior-version facts and winners stale and record a migration note."""
        old_version = self.get_env_version() or "v0"
        if new_version == old_version:
            return {"stale_facts": 0, "stale_winners": 0}
        now = time.time()
        c1 = self.conn.execute(
            "UPDATE facts SET status = 'stale' WHERE status = 'active' AND env_version != ?",
            (new_version,),
        ).rowcount
        c2 = self.conn.execute(
            "UPDATE winners SET status = 'stale' WHERE status = 'active' AND env_version != ?",
            (new_version,),
        ).rowcount
        self.conn.execute(
            "INSERT INTO drift_notes (old_version, new_version, note, created_at) VALUES (?, ?, ?, ?)",
            (old_version, new_version, migration_note.strip() or f"Upgraded {old_version} -> {new_version}", now),
        )
        self.conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('env_version', ?)", (new_version,))
        self.conn.commit()
        return {"stale_facts": int(c1), "stale_winners": int(c2)}

    # ---- Bank 1: Host-verified environment / schema facts -------------------

    def set_fact(self, fact_key: str, value: str, category: str = "schema") -> None:
        env = self.get_env_version() or "v1"
        self.conn.execute(
            """
            INSERT INTO facts (fact_key, category, value, env_version, status, updated_at)
            VALUES (?, ?, ?, ?, 'active', ?)
            ON CONFLICT(fact_key) DO UPDATE SET
                category = excluded.category,
                value = excluded.value,
                env_version = excluded.env_version,
                status = 'active',
                updated_at = excluded.updated_at
            """,
            (fact_key.strip(), category.strip(), value.strip(), env, time.time()),
        )
        self.conn.commit()

    # ---- Bank 2: Verified winners -------------------------------------------

    def record_winner(
        self,
        task_id: str,
        task_text: str,
        solution: str,
        score: float,
        summary: str = "",
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Admit a verified solution; upgrades an existing active entry only if score improves."""
        env = self.get_env_version() or "v1"
        row = self.conn.execute(
            "SELECT score, status FROM winners WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row and row["status"] == "active" and float(row["score"]) >= float(score):
            return False
        self.conn.execute(
            """
            INSERT INTO winners (task_id, task_text, solution, score, summary, tags_json, env_version, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
            ON CONFLICT(task_id) DO UPDATE SET
                task_text = excluded.task_text,
                solution = excluded.solution,
                score = excluded.score,
                summary = excluded.summary,
                tags_json = excluded.tags_json,
                env_version = excluded.env_version,
                status = 'active',
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                task_text.strip(),
                solution.strip(),
                float(score),
                summary.strip(),
                json.dumps(tags or []),
                env,
                time.time(),
            ),
        )
        self.conn.commit()
        return True

    # ---- Bank 3: Host-observed failures & error->repair rules ---------------

    def record_failure(self, task_id: str, error_signature: str, error_detail: str) -> None:
        sig = (error_signature or "unknown_error").strip()[:160]
        env = self.get_env_version() or "v1"
        self.conn.execute(
            "INSERT INTO failure_log (task_id, error_signature, error_detail, env_version, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, sig, (error_detail or "").strip()[:800], env, time.time()),
        )
        self.conn.commit()

    def record_repair(self, error_signature: str, rule: str, fix_example: str = "") -> None:
        sig = error_signature.strip()[:160]
        env = self.get_env_version() or "v1"
        self.conn.execute(
            """
            INSERT INTO repairs (error_signature, rule, fix_example, hits, env_version, status, updated_at)
            VALUES (?, ?, ?, 1, ?, 'active', ?)
            ON CONFLICT(error_signature) DO UPDATE SET
                rule = excluded.rule,
                fix_example = CASE WHEN excluded.fix_example != '' THEN excluded.fix_example ELSE repairs.fix_example END,
                hits = repairs.hits + 1,
                env_version = excluded.env_version,
                status = 'active',
                updated_at = excluded.updated_at
            """,
            (sig, rule.strip()[:400], fix_example.strip()[:600], env, time.time()),
        )
        self.conn.commit()

    def record_outcome(
        self,
        task_id: str,
        task_text: str,
        candidate: str,
        result: VerifyResult,
        recent_error_sig: str = "",
    ) -> dict[str, Any]:
        """Single entry point to update all relevant banks from a host VerifyResult."""
        updated = {"facts": 0, "winner_admitted": False, "repair_linked": False, "failure_logged": False}
        for fk, fv in (result.facts or {}).items():
            self.set_fact(fk, fv)
            updated["facts"] += 1

        if result.ok:
            updated["winner_admitted"] = self.record_winner(
                task_id=task_id,
                task_text=task_text,
                solution=candidate,
                score=result.score,
                summary=result.summary or f"Verified pass (score={result.score:.4g})",
                tags=result.tags,
            )
            if recent_error_sig:
                rule = result.summary or f"Resolved {recent_error_sig} with verified solution for {task_id}."
                self.record_repair(recent_error_sig, rule, fix_example=_compact_block(candidate, 420))
                updated["repair_linked"] = True
        else:
            sig = result.error_signature or _normalize_error_sig(result.error_detail)
            self.record_failure(task_id, sig, result.error_detail)
            updated["failure_logged"] = True
            updated["error_signature"] = sig
        return updated

    # ---- Deterministic retrieval & budget-safe context rendering ------------

    def render_context(
        self,
        task_text: str,
        max_chars: Optional[int] = None,
        max_winners: int = 2,
        max_repairs: int = 5,
        winner_excerpt_chars: int = 900,
    ) -> str:
        """Render active host-verified banks under a hard character cap (never mid-line truncated)."""
        budget = max(512, int(max_chars or self.max_context_chars))
        qtoks = _tokenize(task_text)
        env = self.get_env_version() or "v1"
        sections: list[str] = [f"[VERIFIED_LEDGER env={env}]"]
        used = len(sections[0])

        def try_append(block: str) -> bool:
            nonlocal used
            add = len(block) + 2
            if used + add > budget:
                return False
            sections.append(block)
            used += add
            return True

        # Bank 4: Latest drift migration note (if any)
        drift = self.conn.execute(
            "SELECT old_version, new_version, note FROM drift_notes WHERE new_version = ? ORDER BY id DESC LIMIT 1",
            (env,),
        ).fetchone()
        if drift:
            try_append(f"## Environment Migration ({drift['old_version']} -> {drift['new_version']})\n- {drift['note']}")

        # Bank 1: Active host facts
        facts = self.conn.execute(
            "SELECT category, fact_key, value FROM facts WHERE status = 'active' ORDER BY category, fact_key"
        ).fetchall()
        if facts:
            lines = ["## Verified Environment Facts"]
            for f in facts:
                ln = f"- [{f['category']}] {f['fact_key']}: {f['value']}"
                if used + sum(len(x) + 1 for x in lines) + len(ln) + 4 > budget:
                    break
                lines.append(ln)
            if len(lines) > 1:
                try_append("\n".join(lines))

        # Bank 3: Top relevant error->repair rules
        repairs = self.conn.execute(
            "SELECT error_signature, rule, fix_example, hits FROM repairs WHERE status = 'active' ORDER BY hits DESC, updated_at DESC"
        ).fetchall()
        if repairs and max_repairs > 0:
            scored_r = sorted(
                repairs,
                key=lambda r: (_overlap_score(qtoks, f"{r['error_signature']} {r['rule']}") + 0.05 * min(r["hits"], 10), r["updated_at"] if "updated_at" in r.keys() else 0),
                reverse=True,
            )[:max_repairs]
            lines = ["## Verified Error-Repair Rules"]
            for r in scored_r:
                ln = f"- {r['error_signature']}: {r['rule']}"
                if used + sum(len(x) + 1 for x in lines) + len(ln) + 4 > budget:
                    break
                lines.append(ln)
            if len(lines) > 1:
                try_append("\n".join(lines))

        # Bank 2: Top relevant verified winners
        winners = self.conn.execute(
            "SELECT task_id, task_text, solution, score, summary, tags_json FROM winners WHERE status = 'active'"
        ).fetchall()
        if winners and max_winners > 0:
            scored_w = sorted(
                winners,
                key=lambda w: (_overlap_score(qtoks, f"{w['task_id']} {w['task_text']} {w['summary']} {w['tags_json']}"), float(w["score"])),
                reverse=True,
            )[:max_winners]
            for w in scored_w:
                excerpt = _compact_block(str(w["solution"]), winner_excerpt_chars)
                fence = "~~~~" if "```" in excerpt else "```"
                blk = (
                    f"## Verified Winner ({w['task_id']}, score={float(w['score']):.4g})\n"
                    f"Summary: {w['summary']}\n"
                    f"{fence}\n{excerpt}\n{fence}"
                )
                try_append(blk)

        return "\n\n".join(sections)

    def stats(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "env_version": self.get_env_version(),
            "active_facts": self.conn.execute("SELECT COUNT(*) FROM facts WHERE status = 'active'").fetchone()[0],
            "stale_facts": self.conn.execute("SELECT COUNT(*) FROM facts WHERE status = 'stale'").fetchone()[0],
            "active_winners": self.conn.execute("SELECT COUNT(*) FROM winners WHERE status = 'active'").fetchone()[0],
            "stale_winners": self.conn.execute("SELECT COUNT(*) FROM winners WHERE status = 'stale'").fetchone()[0],
            "active_repairs": self.conn.execute("SELECT COUNT(*) FROM repairs WHERE status = 'active'").fetchone()[0],
            "failures_logged": self.conn.execute("SELECT COUNT(*) FROM failure_log").fetchone()[0],
            "drift_events": self.conn.execute("SELECT COUNT(*) FROM drift_notes").fetchone()[0],
        }


def _normalize_error_sig(error_detail: str) -> str:
    first = (error_detail or "unknown_error").strip().splitlines()[0] if error_detail else "unknown_error"
    first = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", first)
    first = re.sub(r"line \d+", "line N", first)
    if len(first) > 120:
        first = first[:108] + "_" + hashlib.sha1(first.encode()).hexdigest()[:8]
    return first


def call_openai_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 8192,
    temperature: float = 0.2,
    reasoning_effort: Optional[str] = "medium",
    response_format: Optional[dict[str, Any]] = None,
    api_key: str = "none",
    timeout_s: float = 300.0,
) -> tuple[str, str]:
    """Minimal stdlib OpenAI `/v1/chat/completions` client for atlas-ingine / atlas-engine."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    finish_reason = str(choice.get("finish_reason") or "")
    return content, finish_reason


def run_verified_loop(
    ledger: VerifiedLedger,
    task_id: str,
    task_text: str,
    verifier: Callable[[str], VerifyResult],
    generate: Callable[[list[dict[str, Any]]], tuple[str, str]],
    system_prompt: str = "You are a precise engineering assistant. Follow the verified ledger facts and repair rules.",
    max_iters: int = 5,
    stop_on_first_pass: bool = False,
) -> LoopSummary:
    """Run an iterative host-verified hillclimb loop that automatically updates `ledger`."""
    ledger_ctx = ledger.render_context(task_text)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{system_prompt}\n\n{ledger_ctx}"},
        {"role": "user", "content": task_text},
    ]
    best_score = float("-inf")
    best_solution = ""
    solved = False
    last_error_sig = ""
    records: list[dict[str, Any]] = []

    for it in range(1, max_iters + 1):
        candidate, finish_reason = generate(messages)
        vres = verifier(candidate)
        upd = ledger.record_outcome(
            task_id=task_id,
            task_text=task_text,
            candidate=candidate,
            result=vres,
            recent_error_sig=last_error_sig if vres.ok else "",
        )
        records.append(
            {
                "iter": it,
                "ok": vres.ok,
                "score": vres.score,
                "finish_reason": finish_reason,
                "error_signature": vres.error_signature or upd.get("error_signature", ""),
                "ledger_update": upd,
            }
        )
        if vres.ok:
            solved = True
            if vres.score > best_score:
                best_score = vres.score
                best_solution = candidate
            last_error_sig = ""
            if stop_on_first_pass:
                break
            feedback = (
                f"HOST VERIFIER PASSED (score={vres.score:.4g}, best={best_score:.4g}). "
                f"{vres.summary} Improve score further without breaking verification."
            )
        else:
            last_error_sig = vres.error_signature or upd.get("error_signature", "")
            trunc_note = " Note: previous response hit max_tokens; be more concise." if finish_reason == "length" else ""
            feedback = (
                f"HOST VERIFIER FAILED ({last_error_sig}): {vres.error_detail[:600]}{trunc_note}\n"
                "Fix the root cause and return a complete updated solution."
            )

        messages.append({"role": "assistant", "content": candidate})
        messages.append({"role": "user", "content": feedback})

    return LoopSummary(
        task_id=task_id,
        solved=solved,
        best_score=best_score if solved else 0.0,
        best_solution=best_solution,
        iters_used=len(records),
        records=records,
    )


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Inspect and update a SQLite VerifiedLedger.")
    ap.add_argument("--db", required=True, help="Path to ledger SQLite file")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="Print ledger counts and active entries")

    p_fact = sub.add_parser("add-fact", help="Add or update a host-verified environment/schema fact")
    p_fact.add_argument("--key", required=True)
    p_fact.add_argument("--value", required=True)
    p_fact.add_argument("--category", default="schema")

    p_rep = sub.add_parser("add-repair", help="Add or update an error->repair rule")
    p_rep.add_argument("--sig", required=True, help="Host error signature")
    p_rep.add_argument("--rule", required=True, help="Actionable fix rule")
    p_rep.add_argument("--example", default="", help="Optional short fix snippet")

    p_bump = sub.add_parser("bump-env", help="Bump env_version and mark prior facts/winners stale")
    p_bump.add_argument("--version", required=True)
    p_bump.add_argument("--note", default="")

    p_render = sub.add_parser("render", help="Preview rendered ledger context for a task query")
    p_render.add_argument("--query", required=True)
    p_render.add_argument("--max-chars", type=int, default=4000)

    args = ap.parse_args()
    with VerifiedLedger(args.db) as ledger:
        if args.cmd == "inspect":
            print(json.dumps(ledger.stats(), indent=2))
        elif args.cmd == "add-fact":
            ledger.set_fact(args.key, args.value, category=args.category)
            print(json.dumps({"ok": True, "fact_key": args.key, "env_version": ledger.get_env_version()}))
        elif args.cmd == "add-repair":
            ledger.record_repair(args.sig, args.rule, fix_example=args.example)
            print(json.dumps({"ok": True, "error_signature": args.sig}))
        elif args.cmd == "bump-env":
            res = ledger.bump_env_version(args.version, args.note)
            print(json.dumps({"ok": True, "env_version": args.version, **res}))
        elif args.cmd == "render":
            print(ledger.render_context(args.query, max_chars=args.max_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
