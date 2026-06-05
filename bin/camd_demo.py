#!/usr/bin/env python3
"""Thin shim — delegates to the installed entry point.

Prefer the installed CLI command:
    ta-camd-demo [OPTIONS]

Or via uv without installing:
    uv run bin/camd_demo.py [OPTIONS]
"""
from traits_audit._camd_demo import main

if __name__ == "__main__":
    main()
