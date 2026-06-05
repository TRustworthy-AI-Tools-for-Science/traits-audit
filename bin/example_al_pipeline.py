#!/usr/bin/env python3
"""Thin shim — delegates to the installable entry point.

Prefer the installed CLI command:
    ta-demo [OPTIONS]

Or via uv without installing:
    uv run bin/example_al_pipeline.py [OPTIONS]
"""
from traits_audit._example import main

if __name__ == "__main__":
    main()
