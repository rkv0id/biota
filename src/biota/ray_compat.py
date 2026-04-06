# pyright: basic
"""Ray compatibility shim. The one file in the project that runs in pyright basic
mode. All Ray imports and decorators live here so the rest of the codebase can stay
strict. See DECISIONS.md for the rationale.
"""
