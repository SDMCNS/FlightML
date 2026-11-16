"""
Persistent Flight Annotations & Anomaly Notes Engine.

Allows analysts and operations researchers to attach persistent notes, operational
explanations, diplomatic/overflight rights justifications, and external citations to
specific Flight IDs (ectrl_id).

Persisted in `DATA_DIR / flight_notes.json`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import threading

from flightml.config import DATA_DIR
from flightml.data.dataset import FlightDataset

NOTES_FILE = DATA_DIR / "flight_notes.json"
_LOCK = threading.Lock()

NOTE_CATEGORIES: list[str] = [
    "Diplomatic / Overflight Rights",
    "Weather Diversion / Severe Convection",
    "Airspace Closure / Conflict Zone Avoidance",
    "ATC Delay / Severe Airway Capping",
    "Aircraft Technical / Maintenance",
    "Slot Regulation / Curfew Management",
    "Seasonal Transition / Schedule Artifact",
    "Operational Oddity / Unknown",
]

SEED_NOTES: list[dict[str, Any]] = [
    {
        "ectrl_id": 292222833,
        "title": "Lufthansa LH574 - Missing African Overflight Rights (Atlantic Coastal Detour)",
        "category": "Diplomatic / Overflight Rights",
        "author": "Aviation Analyst",
        "tags": ["lufthansa", "lh574", "a350", "overflight-permissions", "seasonal-transition", "cape-town"],
        "source_url": "https://www.aerotelegraph.com/airlines/airbus-a350-von-lufthansa-fliegt-riesigen-umweg-nach-suedafrika/3jft6bs",
        "created_at": "2025-10-28T10:00:00Z",
        "updated_at": "2025-10-28T10:00:00Z",
        "cached_metadata": {
            "period_id": "202510",
            "adep": "EDDM",
            "ades": "FACT",
            "ac_operator": "DLH",
            "ac_type": "A359",
            "ac_reg": "DAIXS",
            "actual_off_block_time": "2025-10-25 18:42:00",
            "actual_distance_nm": 6233,
            "air_time_min": 794,
        },
        "note": (
            "### Summary: Mundane Administrative Overflight Permission Issue\n\n"
            "On **25 October 2025**, Lufthansa flight **LH574** (operated by Airbus A350-900 `D-AIXS`) from Munich (`EDDM`) "
            "to Cape Town (`FACT`) took an extraordinary western routing, taking approximately **13 hours and 14 minutes** "
            "(considerably longer than the normal ~10.5 hour route).\n\n"
            "#### The Normal Route\n"
            "The flight normally takes a relatively direct route south from Munich, crossing the Mediterranean and traveling "
            "through central Africa towards Cape Town.\n\n"
            "#### The 25 October 2025 Routing Taken\n"
            "1. Munich (`EDDM`) across Switzerland (`LSASUIR`), France (`LFFFUIR`), and Spain (`LECMUIR`).\n"
            "2. South past Morocco (`GMMMUIR`) and the Canary Islands (`GCCCUIRN`).\n"
            "3. West African coastline over the Atlantic via Dakar Oceanic (`GOOOUIR` - 249 min dwell).\n"
            "4. Continuing south over the Atlantic via Luanda Oceanic (`FNANUIR`) and Johannesburg Oceanic (`FAJOFIR`).\n"
            "5. Routing briefly through Namibia (`FYWFFIR`) before turning east towards Cape Town (`FACT`).\n\n"
            "#### Actual Cause: Missing Overflight Permissions\n"
            "This was confirmed by Lufthansa to German aviation publication *aeroTELEGRAPH*. The airline's spokesperson stated:\n"
            "> *\"For reasons that are still unclear, the necessary traffic rights were not available.\"*\n\n"
            "#### Operational Context & Theory\n"
            "- **Seasonal Schedule Transition:** 25 October was the first day of LH574's new seasonal operating period. "
            "The IATA winter flight season officially began on 26 October 2025.\n"
            "- **Administrative Theory:** Lufthansa may have failed to secure the required permissions for the 25 October departure "
            "when arranging its new seasonal schedule (administrative / timing coordination issue rather than a weather diversion or aircraft performance limitation).\n"
        ),
    }
]


@dataclass
class FlightNote:
    ectrl_id: int
    title: str
    category: str
    note: str
    author: str = "Analyst"
    tags: list[str] = field(default_factory=list)
    source_url: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cached_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FlightNotesManager:
    """Manager for loading, saving, and querying persistent flight notes."""

    def __init__(self, notes_path: Path | str | None = None) -> None:
        self.notes_path = Path(notes_path) if notes_path else NOTES_FILE
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Create notes file and seed with default notable cases if missing or incomplete."""
        with _LOCK:
            if not self.notes_path.exists():
                self.notes_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_raw(SEED_NOTES)
            else:
                try:
                    data = self._read_raw()
                    # Verify seed notes exist
                    existing_ids = {n["ectrl_id"] for n in data if "ectrl_id" in n}
                    updated = False
                    for s in SEED_NOTES:
                        if s["ectrl_id"] not in existing_ids:
                            data.append(s)
                            updated = True
                    if updated:
                        self._write_raw(data)
                except Exception:
                    self._write_raw(SEED_NOTES)

    def _read_raw(self) -> list[dict[str, Any]]:
        with open(self.notes_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_raw(self, data: list[dict[str, Any]]) -> None:
        tmp_path = self.notes_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.notes_path)

    def list_notes(
        self,
        category: str | None = None,
        search_query: str | None = None,
    ) -> list[FlightNote]:
        """List all notes, optionally filtered by category or keyword search."""
        with _LOCK:
            raw_list = self._read_raw()

        notes = [FlightNote(**r) for r in raw_list]

        if category and category != "All Categories":
            notes = [n for n in notes if n.category.lower() == category.lower()]

        if search_query:
            q = search_query.strip().lower()
            filtered = []
            for n in notes:
                meta = n.cached_metadata or {}
                searchable_text = (
                    f"{n.ectrl_id} {n.title} {n.category} {n.note} {n.author} "
                    f"{' '.join(n.tags)} {meta.get('adep', '')} {meta.get('ades', '')} "
                    f"{meta.get('ac_operator', '')} {meta.get('ac_type', '')} {meta.get('ac_reg', '')}"
                ).lower()
                if q in searchable_text:
                    filtered.append(n)
            notes = filtered

        # Sort by updated_at descending
        notes.sort(key=lambda x: x.updated_at or x.created_at, reverse=True)
        return notes

    def get_note(self, ectrl_id: int) -> FlightNote | None:
        """Get note by ectrl_id."""
        with _LOCK:
            raw_list = self._read_raw()
        for r in raw_list:
            if int(r.get("ectrl_id", 0)) == int(ectrl_id):
                return FlightNote(**r)
        return None

    def save_note(
        self,
        ectrl_id: int,
        title: str,
        note: str,
        category: str = "Operational Oddity / Unknown",
        author: str = "Analyst",
        tags: list[str] | None = None,
        source_url: str | None = None,
        dataset: FlightDataset | None = None,
    ) -> FlightNote:
        """Create or update a note for a flight ID."""
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]

        # Try to resolve metadata from dataset if available
        meta: dict[str, Any] = {}
        if dataset:
            try:
                fl = dataset.get_flight(ectrl_id)
                if fl:
                    meta = {
                        "period_id": str(fl.get("period_id") or dataset.period),
                        "adep": str(fl.get("adep") or ""),
                        "ades": str(fl.get("ades") or ""),
                        "ac_operator": str(fl.get("ac_operator") or ""),
                        "ac_type": str(fl.get("ac_type") or ""),
                        "ac_reg": str(fl.get("ac_reg") or ""),
                        "actual_off_block_time": str(fl.get("actual_off_block_time") or ""),
                        "actual_distance_nm": fl.get("actual_distance_nm"),
                        "air_time_min": fl.get("actual_air_time_min"),
                    }
            except Exception:
                pass

        with _LOCK:
            raw_list = self._read_raw()
            existing_idx = None
            for idx, r in enumerate(raw_list):
                if int(r.get("ectrl_id", 0)) == int(ectrl_id):
                    existing_idx = idx
                    break

            if existing_idx is not None:
                orig = raw_list[existing_idx]
                created_at = orig.get("created_at", now_iso)
                # Keep existing cached_metadata if new query didn't find anything
                if not meta and orig.get("cached_metadata"):
                    meta = orig.get("cached_metadata", {})
                new_note = FlightNote(
                    ectrl_id=int(ectrl_id),
                    title=title.strip(),
                    category=category,
                    note=note.strip(),
                    author=author.strip() or "Analyst",
                    tags=clean_tags,
                    source_url=source_url.strip() if source_url else None,
                    created_at=created_at,
                    updated_at=now_iso,
                    cached_metadata=meta,
                )
                raw_list[existing_idx] = new_note.to_dict()
            else:
                new_note = FlightNote(
                    ectrl_id=int(ectrl_id),
                    title=title.strip(),
                    category=category,
                    note=note.strip(),
                    author=author.strip() or "Analyst",
                    tags=clean_tags,
                    source_url=source_url.strip() if source_url else None,
                    created_at=now_iso,
                    updated_at=now_iso,
                    cached_metadata=meta,
                )
                raw_list.append(new_note.to_dict())

            self._write_raw(raw_list)

        return new_note

    def delete_note(self, ectrl_id: int) -> bool:
        """Delete a note by ectrl_id."""
        with _LOCK:
            raw_list = self._read_raw()
            new_list = [r for r in raw_list if int(r.get("ectrl_id", 0)) != int(ectrl_id)]
            deleted = len(new_list) < len(raw_list)
            if deleted:
                self._write_raw(new_list)
        return deleted


# Global singleton instance
_DEFAULT_MANAGER: FlightNotesManager | None = None


def get_notes_manager() -> FlightNotesManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = FlightNotesManager()
    return _DEFAULT_MANAGER
