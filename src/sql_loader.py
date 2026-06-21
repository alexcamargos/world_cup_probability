"""Load SQL files and lightweight SQL templates used by the pipeline."""

from __future__ import annotations

import re
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent / "sql"
_TEMPLATE_VARIABLE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def read_sql(relative_path: str) -> str:
    """Read a SQL file from ``src/sql``."""
    return _sql_path(relative_path).read_text(encoding="utf-8")


def render_sql_template(relative_path: str, **context: object) -> str:
    """Render a SQL template using simple ``{{ variable }}`` substitution."""
    template = read_sql(relative_path)
    variables = set(_TEMPLATE_VARIABLE.findall(template))
    missing = sorted(variables.difference(context))
    if missing:
        raise KeyError(f"Missing SQL template variables for {relative_path}: {', '.join(missing)}")

    return _TEMPLATE_VARIABLE.sub(lambda match: str(context[match.group(1)]), template)


def _sql_path(relative_path: str) -> Path:
    path = SQL_DIR / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path
