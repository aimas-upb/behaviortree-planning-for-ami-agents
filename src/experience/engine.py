"""
Experience Management Engine.

Stores and retrieves mappings from structured intents to BT leaf-node
JSON-IR representations.  Supports persistence via JSON and maintains
an in-memory slot-key index for fast retrieval.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .intent import StructuredIntent

logger = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """A single stored experience from a previously successful execution."""

    id: str
    text_intent: str
    affordance_type: str
    artifact_type: str
    workspace_type: str
    verb: str
    bt_leaf_json_ir: dict  # JSON-IR of the BT leaf node (empty if infeasible)
    is_infeasible: bool = False
    home_id: str = (
        ""  # home the experience was recorded in (used to scope infeasible matches)
    )
    embedding: Optional[list[float]] = None
    created_at: str = ""
    source_test_id: str = ""

    def slot_key(self) -> tuple:
        """Return the slot-matching key."""
        return (self.affordance_type, self.artifact_type, self.workspace_type)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text_intent": self.text_intent,
            "affordance_type": self.affordance_type,
            "artifact_type": self.artifact_type,
            "workspace_type": self.workspace_type,
            "verb": self.verb,
            "bt_leaf_json_ir": self.bt_leaf_json_ir,
            "is_infeasible": self.is_infeasible,
            "home_id": self.home_id,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "source_test_id": self.source_test_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceEntry":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text_intent=data.get("text_intent", ""),
            affordance_type=data.get("affordance_type", ""),
            artifact_type=data.get("artifact_type", ""),
            workspace_type=data.get("workspace_type", ""),
            verb=data.get("verb", "set"),
            bt_leaf_json_ir=data.get("bt_leaf_json_ir", {}),
            is_infeasible=data.get("is_infeasible", False),
            home_id=data.get("home_id", ""),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", ""),
            source_test_id=data.get("source_test_id", ""),
        )


class ExperienceEngine:
    """
    Persistent storage for execution experiences.

    Each experience entry maps a structured intent to the JSON-IR of the
    BT leaf node that successfully addressed it (or marks the intent as
    infeasible).  The engine maintains a slot-key index for fast
    candidate retrieval during matching.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self.entries: list[ExperienceEntry] = []
        self._slot_index: dict[tuple, list[int]] = {}
        if persistence_path:
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load experiences from the JSON persistence file."""
        if not self.persistence_path:
            return
        path = Path(self.persistence_path)
        if not path.exists():
            logger.info(f"No existing experience store at {path}")
            return
        try:
            with open(path) as f:
                data = json.load(f)
            raw_entries = data.get("entries", [])
            for raw in raw_entries:
                entry = ExperienceEntry.from_dict(raw)
                idx = len(self.entries)
                self.entries.append(entry)
                key = entry.slot_key()
                self._slot_index.setdefault(key, []).append(idx)
            logger.info(f"Loaded {len(self.entries)} experiences from {path}")
        except Exception as e:
            logger.error(f"Failed to load experience store: {e}")

    def save(self) -> None:
        """Persist all experiences to the JSON file."""
        if not self.persistence_path:
            return
        path = Path(self.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(self.entries)} experiences to {path}")

    # ------------------------------------------------------------------
    # Add / query
    # ------------------------------------------------------------------

    def add(self, entry: ExperienceEntry) -> None:
        """Add a new experience entry and update the slot index."""
        idx = len(self.entries)
        self.entries.append(entry)
        key = entry.slot_key()
        self._slot_index.setdefault(key, []).append(idx)
        logger.debug(
            f"Added experience {entry.id}: {entry.text_intent!r} "
            f"(slot={key}, infeasible={entry.is_infeasible})"
        )

    def add_infeasible(
        self,
        intent: StructuredIntent,
        source_test_id: str,
        home_id: str = "",
    ) -> None:
        """Store an infeasible intent (no BT leaf node)."""
        entry = ExperienceEntry(
            id=str(uuid.uuid4()),
            text_intent=intent.text_intent,
            affordance_type=intent.affordance_type,
            artifact_type=intent.artifact_type,
            workspace_type=intent.workspace_type,
            verb=intent.verb,
            bt_leaf_json_ir={},
            is_infeasible=True,
            home_id=home_id,
            created_at=datetime.now().isoformat(),
            source_test_id=source_test_id,
        )
        self.add(entry)

    def get_by_slot_key(self, slot_key: tuple) -> list[ExperienceEntry]:
        """Get all experiences matching a slot key."""
        indices = self._slot_index.get(slot_key, [])
        return [self.entries[i] for i in indices]

    def get_all(self) -> list[ExperienceEntry]:
        """Return all stored entries."""
        return list(self.entries)

    def has_identical(self, intent: StructuredIntent) -> bool:
        """
        Check if an experience with the same text_intent *and* slot_key
        already exists.
        """
        key = intent.slot_key()
        for entry in self.get_by_slot_key(key):
            if entry.text_intent == intent.text_intent:
                return True
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        """Remove all entries and reset the index."""
        self.entries.clear()
        self._slot_index.clear()
        logger.info("Experience engine cleared")

    def stats(self) -> dict:
        feasible = sum(1 for e in self.entries if not e.is_infeasible)
        infeasible = sum(1 for e in self.entries if e.is_infeasible)
        unique_slots = len(self._slot_index)
        return {
            "total_entries": len(self.entries),
            "feasible_entries": feasible,
            "infeasible_entries": infeasible,
            "unique_slot_keys": unique_slots,
        }
