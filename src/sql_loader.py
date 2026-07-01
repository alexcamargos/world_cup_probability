from __future__ import annotations

from pathlib import Path

import jinja2
import jinja2.meta

SQL_DIR = Path(__file__).resolve().parent / "sql"


def read_sql(relative_path: str) -> str:
    """Read a SQL file from ``src/sql``.

    Args:
        relative_path: The relative path to the SQL file from ``src/sql``.

    Returns:
        The content of the SQL file.
    """
    return _sql_path(relative_path).read_text(encoding="utf-8")


def render_sql_template(relative_path: str, **context: object) -> str:
    """Render a SQL template using Jinja2.

    Args:
        relative_path: The relative path to the SQL template from ``src/sql``.
        **context: The template variables to render.

    Returns:
        The rendered SQL query string.

    Raises:
        KeyError: If any required template variables are missing.
    """
    template_content = read_sql(relative_path)
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    ast = env.parse(template_content)
    variables = jinja2.meta.find_undeclared_variables(ast)
    missing = sorted(variables.difference(context))
    if missing:
        raise KeyError(
            f"Missing SQL template variables for {relative_path}: {', '.join(missing)}"
        )

    template = env.from_string(template_content)
    return template.render(**context)


def _sql_path(relative_path: str) -> Path:
    """Get the absolute path to a SQL file in ``src/sql``.

    Args:
        relative_path: The relative path to the SQL file.

    Returns:
        The absolute path to the SQL file.

    Raises:
        FileNotFoundError: If the SQL file does not exist.
    """
    path = SQL_DIR / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path

