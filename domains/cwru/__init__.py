"""
domains/cwru
============
CWRU Bearing Fault Detection domain plugin for NeuroVerse.

Wraps the Case Western Reserve University (CWRU) bearing fault dataset:
  normal     — healthy bearing baseline
  inner_race — fault on the inner race
  ball       — fault on the rolling element (ball)
  outer_race — fault on the outer race

Usage
-----
    from domains.cwru.domain import CWRUDomain
    domain = CWRUDomain()
"""

from .domain import CWRUDomain  # noqa: F401

__all__ = ["CWRUDomain"]
