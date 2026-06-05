#!/usr/bin/env python3
"""Thin shim — delegates to the installed entry point.

Prefer the installed CLI command:
    ta-sdl-demo [OPTIONS]

Or via uv without installing:
    uv run bin/sdl_demo.py [OPTIONS]
"""
from traits_audit._sdl_demo import main

if __name__ == "__main__":
    main()
