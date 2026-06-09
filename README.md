# find-alan

Python package skeleton managed with `uv`.

## Setup

Install the package environment:

```sh
uv sync
```

Run the starter script:

```sh
uv run find-alan
```

Add runnable scripts by creating modules under `src/find_alan/scripts/` with a `main()` function, then adding them to `[project.scripts]` in `pyproject.toml`.

Build the package:

```sh
uv build
```
