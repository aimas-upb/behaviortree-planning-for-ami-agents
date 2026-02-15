"""
Step 2: Experience Matching.

Matches structured intents against stored experiences using a hybrid
approach:
  1. Sentence-embedding similarity (BAAI/bge-small-en-v1.5 via FastEmbed)
     filters candidates with cosine similarity >= threshold.
  2. Slot matching requires an exact match on (affordance_type,
     artifact_type, workspace_type).
  3. Among perfect slot matches above the similarity threshold, the one
     with the highest embedding similarity wins.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from .engine import ExperienceEngine, ExperienceEntry
from .intent import StructuredIntent

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a single intent against the experience engine."""

    intent: StructuredIntent
    matched: bool = False
    experience: Optional[ExperienceEntry] = None
    similarity_score: float = 0.0
    is_infeasible: bool = False

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "matched": self.matched,
            "experience_id": self.experience.id if self.experience else None,
            "similarity_score": self.similarity_score,
            "is_infeasible": self.is_infeasible,
        }


class ExperienceMatcher:
    """
    Matches structured intents against stored experiences.

    Uses a two-stage matching strategy:
      1. Cosine similarity on ``text_intent`` embeddings (threshold filter).
      2. Perfect slot-key match among candidates above the threshold.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.85,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Lazy-load the FastEmbed text embedding model."""
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info(f"Loading embedding model: {self.embedding_model}")
            self._model = TextEmbedding(model_name=self.embedding_model)
            logger.info("Embedding model loaded")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Ensure cached embeddings
    # ------------------------------------------------------------------

    def ensure_embeddings(self, engine: ExperienceEngine) -> None:
        """
        Pre-compute and cache embeddings for any engine entries that
        do not yet have one.
        """
        missing = [
            (i, e) for i, e in enumerate(engine.entries) if e.embedding is None
        ]
        if not missing:
            return
        texts = [e.text_intent for _, e in missing]
        logger.info(f"Computing embeddings for {len(texts)} entries")
        embeddings = self.embed(texts)
        for (i, entry), emb in zip(missing, embeddings):
            entry.embedding = emb

    # ------------------------------------------------------------------
    # Match
    # ------------------------------------------------------------------

    def match(
        self,
        intents: list[StructuredIntent],
        engine: ExperienceEngine,
    ) -> list[MatchResult]:
        """
        Match each intent against stored experiences.

        For each intent:
          1. Compute the embedding for its ``text_intent``.
          2. Compare against *all* entries in the engine.
          3. Filter to entries with cosine similarity >= threshold.
          4. Among those, find entries with a perfect slot-key match.
          5. Return the best (highest similarity) perfect-slot match,
             or no match if none qualifies.

        Returns one ``MatchResult`` per intent (same order as input).
        """
        if engine.size() == 0:
            logger.info("Experience engine is empty — no matches possible")
            return [MatchResult(intent=intent) for intent in intents]

        # Ensure all stored entries have embeddings
        self.ensure_embeddings(engine)

        # Embed all incoming intents in a single batch
        intent_texts = [i.text_intent for i in intents]
        intent_embeddings = self.embed(intent_texts)

        results: list[MatchResult] = []

        for intent, intent_emb in zip(intents, intent_embeddings):
            best_entry: Optional[ExperienceEntry] = None
            best_sim: float = 0.0
            intent_slot = intent.slot_key()

            for entry in engine.entries:
                if entry.embedding is None:
                    continue

                sim = self.cosine_similarity(intent_emb, entry.embedding)
                if sim < self.similarity_threshold:
                    continue

                # Check perfect slot match
                if entry.slot_key() != intent_slot:
                    continue

                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None:
                results.append(
                    MatchResult(
                        intent=intent,
                        matched=True,
                        experience=best_entry,
                        similarity_score=best_sim,
                        is_infeasible=best_entry.is_infeasible,
                    )
                )
                logger.info(
                    f"Matched intent {intent.text_intent!r} "
                    f"-> experience {best_entry.id} "
                    f"(sim={best_sim:.3f}, infeasible={best_entry.is_infeasible})"
                )
            else:
                results.append(MatchResult(intent=intent))
                logger.info(
                    f"No match for intent {intent.text_intent!r} "
                    f"(slot={intent_slot})"
                )

        matched_count = sum(1 for r in results if r.matched)
        logger.info(
            f"Matching complete: {matched_count}/{len(intents)} intents matched "
            f"(engine size={engine.size()})"
        )

        return results
