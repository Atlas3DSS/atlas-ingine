"""Standalone Celebration Video Ledger (`celebration_video_ledger.sqlite`) with Addition AND Subtraction.

Every entry has an explicit lifecycle (`active`, `superseded`, `retracted`) and every
state transition is logged in `ledger_events` for full auditability:
  - ADDITION: Host-gated admission of verified scene techniques, camera/lighting/shader
    recipes, aesthetic lessons, and crash->fix repair rules. Unverified claims from
    failing iterations are rejected.
  - SUBTRACTION / PRUNING / SUPERSEDING:
      * `retract_entry`: Explicitly subtracts an active rule/recipe that caused a visual
        defect, regression, or conflict (marks `status='retracted'`, logs `RETRACT`, and
        keeps a compact one-line caution so the model does not repeat the bad pattern).
      * `supersede_entry`: When a higher-scoring verified winner arrives for the same
        `slot_key` (e.g., `lighting_rig`, `camera_choreography`, `material_palette`),
        marks the previous winner `status='superseded'` and promotes the new entry.
      * `prune_to_budget`: Demotes lowest-scoring active entries per category if active
        count exceeds budget so Tier 1 stays razor-sharp across long hillclimbs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any


VALID_CATEGORIES = (
    "api_fact",
    "scene_technique",
    "repair_rule",
    "aesthetic_lesson",
)

VALID_STATUSES = ("active", "superseded", "retracted")


@dataclass(frozen=True)
class LedgerEntry:
    entry_id: int
    category: str
    slot_key: str
    title: str
    summary: str
    snippet: str
    status: str
    source_iter: int
    source_score: float
    host_verified: bool
    vision_verified: bool
    reason: str
    superseded_by: int | None


class CelebrationVideoLedger:
    """SQLite-backed standalone ledger supporting verified Addition and audited Subtraction."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                slot_key TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                snippet TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                blender_fp TEXT NOT NULL DEFAULT '',
                source_iter INTEGER NOT NULL DEFAULT 0,
                source_score REAL NOT NULL DEFAULT 0.0,
                host_verified INTEGER NOT NULL DEFAULT 0,
                vision_verified INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                superseded_by INTEGER DEFAULT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_status_cat ON entries(status, category);
            CREATE INDEX IF NOT EXISTS idx_entries_slot ON entries(slot_key, status);

            CREATE TABLE IF NOT EXISTS ledger_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                iter_num INTEGER NOT NULL,
                action TEXT NOT NULL,
                entry_id INTEGER,
                target_id INTEGER,
                slot_key TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                meta_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _log_event(
        self,
        *,
        iter_num: int,
        action: str,
        entry_id: int | None = None,
        target_id: int | None = None,
        slot_key: str = "",
        reason: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO ledger_events(ts, iter_num, action, entry_id, target_id, slot_key, reason, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                int(iter_num),
                str(action),
                entry_id,
                target_id,
                str(slot_key),
                str(reason),
                json.dumps(meta or {}, sort_keys=True),
            ),
        )

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
        return LedgerEntry(
            entry_id=int(row["entry_id"]),
            category=str(row["category"]),
            slot_key=str(row["slot_key"]),
            title=str(row["title"]),
            summary=str(row["summary"]),
            snippet=str(row["snippet"]),
            status=str(row["status"]),
            source_iter=int(row["source_iter"]),
            source_score=float(row["source_score"]),
            host_verified=bool(row["host_verified"]),
            vision_verified=bool(row["vision_verified"]),
            reason=str(row["reason"]),
            superseded_by=int(row["superseded_by"]) if row["superseded_by"] is not None else None,
        )

    # ------------------------------------------------------------------
    # VERIFIED ADDITION (with automatic superseding on same slot_key)
    # ------------------------------------------------------------------
    def admit_entry(
        self,
        *,
        category: str,
        slot_key: str,
        title: str,
        summary: str,
        snippet: str = "",
        iter_num: int = 0,
        score: float = 0.0,
        host_verified: bool = False,
        vision_verified: bool = False,
        blender_fp: str = "",
        reason: str = "",
        supersede_same_slot: bool = True,
    ) -> tuple[int | None, list[int]]:
        """Admit a new verified entry. Returns (new_entry_id_or_None, superseded_ids).

        Admission gate:
          - `api_fact` & `repair_rule`: require `host_verified=True`.
          - `scene_technique` & `aesthetic_lesson`: require BOTH `host_verified=True` and `vision_verified=True`.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category {category!r}; expected one of {VALID_CATEGORIES}")
        slot_key = slot_key.strip()[:80]
        title = title.strip()[:120]
        summary = summary.strip()[:420]
        snippet = snippet.strip()[:1400]

        if not title or not summary:
            return None, []

        if category in ("api_fact", "repair_rule"):
            if not host_verified:
                self._log_event(
                    iter_num=iter_num,
                    action="REJECT_UNVERIFIED",
                    slot_key=slot_key,
                    reason=f"{category} requires host_verified=True",
                    meta={"title": title},
                )
                self._conn.commit()
                return None, []
        else:
            if not (host_verified and vision_verified):
                self._log_event(
                    iter_num=iter_num,
                    action="REJECT_UNVERIFIED",
                    slot_key=slot_key,
                    reason=f"{category} requires host_verified=True AND vision_verified=True",
                    meta={"title": title, "host_verified": host_verified, "vision_verified": vision_verified},
                )
                self._conn.commit()
                return None, []

        # Deduplicate exact active title+summary in same slot
        existing_same = self._conn.execute(
            """
            SELECT entry_id, source_score FROM entries
            WHERE status = 'active' AND category = ? AND slot_key = ? AND title = ? AND summary = ?
            """,
            (category, slot_key, title, summary),
        ).fetchone()
        if existing_same is not None:
            return int(existing_same["entry_id"]), []

        # If superseding same slot, only supersede if new score >= max active score in that slot (or api_fact refresh)
        prior_active = self._conn.execute(
            """
            SELECT entry_id, source_score, title FROM entries
            WHERE status = 'active' AND category = ? AND slot_key = ?
            ORDER BY source_score DESC, entry_id DESC
            """,
            (category, slot_key),
        ).fetchall()

        if supersede_same_slot and prior_active and category in ("scene_technique", "aesthetic_lesson"):
            best_prior = float(prior_active[0]["source_score"])
            if float(score) < best_prior:
                self._log_event(
                    iter_num=iter_num,
                    action="SKIP_LOWER_SCORE",
                    target_id=int(prior_active[0]["entry_id"]),
                    slot_key=slot_key,
                    reason=f"candidate score {score:.2f} < active winner {best_prior:.2f}",
                    meta={"title": title},
                )
                self._conn.commit()
                return None, []

        now = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO entries(
                category, slot_key, title, summary, snippet, status, blender_fp,
                source_iter, source_score, host_verified, vision_verified, reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                category,
                slot_key,
                title,
                summary,
                snippet,
                blender_fp,
                int(iter_num),
                float(score),
                int(bool(host_verified)),
                int(bool(vision_verified)),
                reason.strip()[:300],
                now,
                now,
            ),
        )
        new_id = int(cur.lastrowid)
        self._log_event(
            iter_num=iter_num,
            action="ADD",
            entry_id=new_id,
            slot_key=slot_key,
            reason=reason or f"verified addition at iter {iter_num} (score={score:.1f})",
            meta={"category": category, "title": title, "score": score},
        )

        superseded_ids: list[int] = []
        if supersede_same_slot and prior_active:
            for row in prior_active:
                old_id = int(row["entry_id"])
                sup_reason = reason or f"superseded by #{new_id} at iter {iter_num} (score {score:.1f} >= {row['source_score']:.1f})"
                self._conn.execute(
                    """
                    UPDATE entries
                    SET status = 'superseded', superseded_by = ?, reason = ?, updated_at = ?
                    WHERE entry_id = ?
                    """,
                    (new_id, sup_reason[:300], now, old_id),
                )
                self._log_event(
                    iter_num=iter_num,
                    action="SUPERSEDE",
                    entry_id=old_id,
                    target_id=new_id,
                    slot_key=slot_key,
                    reason=sup_reason,
                    meta={"old_title": row["title"], "new_title": title},
                )
                superseded_ids.append(old_id)

        self._conn.commit()
        return new_id, superseded_ids

    # ------------------------------------------------------------------
    # EXPLICIT SUBTRACTION / RETRACTION / PRUNING
    # ------------------------------------------------------------------
    def retract_entry(
        self,
        *,
        entry_id: int | None = None,
        slot_key: str | None = None,
        iter_num: int,
        reason: str,
    ) -> list[int]:
        """Explicitly subtract active entry/entries that caused a defect or conflict.

        Preserves full audit trail (`status='retracted'`) and logs `RETRACT` event(s).
        """
        if entry_id is None and not slot_key:
            raise ValueError("Must provide entry_id or slot_key to retract")
        if entry_id is not None:
            rows = self._conn.execute(
                "SELECT entry_id, category, slot_key, title FROM entries WHERE status = 'active' AND entry_id = ?",
                (int(entry_id),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT entry_id, category, slot_key, title FROM entries WHERE status = 'active' AND slot_key = ?",
                (str(slot_key).strip(),),
            ).fetchall()

        now = time.time()
        retracted: list[int] = []
        clean_reason = (reason or "retracted after visual/host defect").strip()[:300]
        for row in rows:
            eid = int(row["entry_id"])
            # Never retract pinned Blender host API facts unless explicitly targeted by version drift
            if row["category"] == "api_fact" and entry_id is None:
                continue
            self._conn.execute(
                "UPDATE entries SET status = 'retracted', reason = ?, updated_at = ? WHERE entry_id = ?",
                (clean_reason, now, eid),
            )
            self._log_event(
                iter_num=iter_num,
                action="RETRACT",
                entry_id=eid,
                slot_key=str(row["slot_key"]),
                reason=clean_reason,
                meta={"category": row["category"], "title": row["title"]},
            )
            retracted.append(eid)
        self._conn.commit()
        return retracted

    def invalidate_api_facts_on_drift(self, new_blender_fp: str, *, iter_num: int = 0) -> int:
        """Retract stale Blender API facts if execution binary fingerprint changed."""
        rows = self._conn.execute(
            "SELECT entry_id, slot_key FROM entries WHERE status = 'active' AND category = 'api_fact' AND blender_fp != '' AND blender_fp != ?",
            (new_blender_fp,),
        ).fetchall()
        now = time.time()
        for row in rows:
            eid = int(row["entry_id"])
            self._conn.execute(
                "UPDATE entries SET status = 'retracted', reason = ?, updated_at = ? WHERE entry_id = ?",
                (f"Blender fingerprint drifted to {new_blender_fp}", now, eid),
            )
            self._log_event(
                iter_num=iter_num,
                action="RETRACT_DRIFT",
                entry_id=eid,
                slot_key=str(row["slot_key"]),
                reason=f"Blender fingerprint drifted to {new_blender_fp}",
            )
        self._conn.commit()
        return len(rows)

    def prune_to_budget(
        self,
        *,
        iter_num: int,
        max_per_category: dict[str, int] | None = None,
    ) -> list[int]:
        """Keep active ledger sharp by superseding/retracting overflow lowest-score entries."""
        caps = max_per_category or {
            "api_fact": 12,
            "scene_technique": 6,
            "repair_rule": 6,
            "aesthetic_lesson": 5,
        }
        pruned: list[int] = []
        now = time.time()
        for cat, cap in caps.items():
            rows = self._conn.execute(
                """
                SELECT entry_id, slot_key, title, source_score FROM entries
                WHERE status = 'active' AND category = ?
                ORDER BY source_score DESC, source_iter DESC, entry_id DESC
                """,
                (cat,),
            ).fetchall()
            if len(rows) <= cap:
                continue
            for overflow in rows[cap:]:
                eid = int(overflow["entry_id"])
                reason = f"pruned to active budget cap ({cap}) for {cat}"
                self._conn.execute(
                    "UPDATE entries SET status = 'superseded', reason = ?, updated_at = ? WHERE entry_id = ?",
                    (reason, now, eid),
                )
                self._log_event(
                    iter_num=iter_num,
                    action="PRUNE",
                    entry_id=eid,
                    slot_key=str(overflow["slot_key"]),
                    reason=reason,
                    meta={"category": cat, "title": overflow["title"], "score": overflow["source_score"]},
                )
                pruned.append(eid)
        self._conn.commit()
        return pruned

    # ------------------------------------------------------------------
    # QUERIES & COMPACT TIER-1 RENDERING
    # ------------------------------------------------------------------
    def list_entries(self, *, status: str | None = "active", category: str | None = None) -> list[LedgerEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM entries {where} ORDER BY category, source_score DESC, entry_id DESC",
            params,
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def recent_retractions(self, limit: int = 5) -> list[LedgerEntry]:
        rows = self._conn.execute(
            """
            SELECT * FROM entries
            WHERE status = 'retracted'
            ORDER BY updated_at DESC, entry_id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        by_status = dict(
            self._conn.execute("SELECT status, COUNT(*) FROM entries GROUP BY status").fetchall()
        )
        by_action = dict(
            self._conn.execute("SELECT action, COUNT(*) FROM ledger_events GROUP BY action").fetchall()
        )
        active_by_cat = dict(
            self._conn.execute("SELECT category, COUNT(*) FROM entries WHERE status = 'active' GROUP BY category").fetchall()
        )
        return {
            "entries_by_status": {k: int(v) for k, v in by_status.items()},
            "active_by_category": {k: int(v) for k, v in active_by_cat.items()},
            "events_by_action": {k: int(v) for k, v in by_action.items()},
        }

    def render_tier1_ledger(self, *, max_chars: int = 6500) -> str:
        """Render the compact active ledger + recent retractions for Tier 1 pinned context."""
        sections: list[str] = []

        def _fmt_group(header: str, entries: list[LedgerEntry], include_snippet: bool) -> str:
            if not entries:
                return ""
            lines = [f"### {header}"]
            for e in entries:
                tag = f"[#{e.entry_id} slot={e.slot_key} iter={e.source_iter} score={e.source_score:.1f}]"
                lines.append(f"- {tag} **{e.title}**: {e.summary}")
                if include_snippet and e.snippet:
                    sn = e.snippet.strip()
                    if len(sn) > 360:
                        sn = sn[:357].rstrip() + "..."
                    lines.append(f"  ```python\n  {sn.replace(chr(10), chr(10) + '  ')}\n  ```")
            return "\n".join(lines)

        api_facts = self.list_entries(status="active", category="api_fact")
        techniques = self.list_entries(status="active", category="scene_technique")
        lessons = self.list_entries(status="active", category="aesthetic_lesson")
        repairs = self.list_entries(status="active", category="repair_rule")
        retracted = self.recent_retractions(limit=5)

        if api_facts:
            sections.append(_fmt_group("Active Host-Verified Blender 4.5 API Truths", api_facts[:10], False))
        if techniques:
            sections.append(_fmt_group("Active Verified Scene & Craft Winners (ADDITIONS)", techniques[:6], True))
        if lessons:
            sections.append(_fmt_group("Active Vision-Verified Aesthetic Lessons", lessons[:5], False))
        if repairs:
            sections.append(_fmt_group("Active Verified Error->Fix Guards", repairs[:6], True))
        if retracted:
            rlines = ["### Recently Subtracted / Retracted Patterns (DO NOT REPEAT)"]
            for e in retracted:
                rlines.append(f"- [RETRACTED #{e.entry_id} slot={e.slot_key}] {e.title} — WHY: {e.reason}")
            sections.append("\n".join(rlines))

        if not sections:
            return "[CELEBRATION_VIDEO_LEDGER: empty — seed first verified techniques from Iteration 1]"

        body = "[CELEBRATION_VIDEO_LEDGER (active verified memory; audited add/subtract)]\n" + "\n\n".join(sections)
        if len(body) > max_chars:
            # Trim snippets first before ever truncating lines
            compact_sections: list[str] = []
            if api_facts:
                compact_sections.append(_fmt_group("Active Host-Verified Blender 4.5 API Truths", api_facts[:8], False))
            if techniques:
                compact_sections.append(_fmt_group("Active Verified Scene & Craft Winners", techniques[:6], False))
            if lessons:
                compact_sections.append(_fmt_group("Active Vision-Verified Aesthetic Lessons", lessons[:5], False))
            if repairs:
                compact_sections.append(_fmt_group("Active Verified Error->Fix Guards", repairs[:5], False))
            if retracted:
                rlines = ["### Recently Subtracted / Retracted Patterns (DO NOT REPEAT)"]
                for e in retracted:
                    rlines.append(f"- [RETRACTED #{e.entry_id} slot={e.slot_key}] {e.title} — WHY: {e.reason}")
                compact_sections.append("\n".join(rlines))
            body = "[CELEBRATION_VIDEO_LEDGER (compact)]\n" + "\n\n".join(compact_sections)
            if len(body) > max_chars:
                body = body[: max_chars - 24].rstrip() + "\n...(ledger capped)"
        return body
