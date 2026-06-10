"""Compatibility wrapper for the packaged tiled refinement CLI."""

from find_alan.scripts.refine import main


if __name__ == "__main__":
    raise SystemExit(main())
