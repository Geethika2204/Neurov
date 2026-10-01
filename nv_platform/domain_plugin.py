"""
/platform/domain_plugin.py

Abstract base class that every NeuroVerse domain must subclass.

Contract
--------
A domain supplies four pieces of metadata and optionally overrides
``validate_event``.  The platform layers never import from ``/domains/``;
they only call the methods defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jsonschema


class DomainPlugin(ABC):
    """
    Abstract contract for every NeuroVerse domain.

    Subclass this in ``/domains/<name>/domain.py`` and implement all four
    abstract properties.  The platform layers depend only on this interface —
    they never import concrete domain classes.

    Example
    -------
    >>> class MyDomain(DomainPlugin):
    ...     @property
    ...     def input_schema(self):
    ...         return {"type": "object", "properties": {"value": {"type": "number"}}}
    ...     @property
    ...     def output_schema(self):
    ...         return {"type": "object", "properties": {"label": {"type": "string"}}}
    ...     @property
    ...     def knowledge_base_path(self):
    ...         return "domains/my_domain/rag_kb"
    ...     @property
    ...     def threshold_hints(self):
    ...         return {"novelty": 0.6, "uncertainty": 0.7}
    """

    # ------------------------------------------------------------------
    # Abstract properties — every domain must define these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """
        JSON Schema (draft-7) describing a single sensor event dict.

        The schema is used by ``validate_event`` and by the platform's
        type-checking machinery.  Example::

            {
                "type": "object",
                "required": ["timestamp", "value"],
                "properties": {
                    "timestamp": {"type": "number"},
                    "value":     {"type": "number"},
                }
            }
        """

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, Any]:
        """
        JSON Schema describing the agent decision dict returned for this domain.
        """

    @property
    @abstractmethod
    def knowledge_base_path(self) -> str:
        """
        Path to the ChromaDB persist directory for this domain's RAG knowledge base.
        Relative to the repo root or absolute.
        """

    @property
    @abstractmethod
    def threshold_hints(self) -> dict[str, float]:
        """
        Soft guidance for the gate learner's initial decision boundary.

        Keys correspond to ``ReasoningScore`` field names
        (``"novelty"``, ``"uncertainty"``, ``"prediction_error"``,
        ``"memory_confidence"``).  Values are floats in [0, 1].

        These are *hints*, not hard thresholds — the gate learner will adapt
        them over time.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — domains may override
    # ------------------------------------------------------------------

    def validate_event(self, event: dict[str, Any]) -> bool:
        """
        Validate *event* against ``self.input_schema``.

        Returns ``True`` if valid, ``False`` if not.  Logs a warning on
        schema violation so callers can decide whether to drop the event.
        """
        try:
            jsonschema.validate(instance=event, schema=self.input_schema)
            return True
        except jsonschema.ValidationError:
            return False

    def domain_name(self) -> str:
        """Human-readable name derived from the class name."""
        return type(self).__name__
