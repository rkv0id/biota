"""Shared visualization utilities for biota.

This package holds visualization helpers used by standalone scripts
(view_archive.py, render_creature.py, render_rollout.py) and eventually
by the dashboard module. The scope is "given a mass field or an archive,
produce something a human can look at" - not the search or simulation
logic, and not the dashboard server itself.

Current modules:
- colormap: grayscale-to-RGB lookup tables (magma, etc) for rendering
  mass fields with visible structure.
- render: archive page renderer.  Converts a biota Archive into a
  self-contained HTML string.  Used by view_archive.py (per-run CLI)
  and build_index.py (batch index builder).
"""
