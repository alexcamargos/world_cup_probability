"""Tests for the SQL loader module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.sql_loader import read_sql, render_sql_template

if TYPE_CHECKING:
    from pathlib import Path


def test_read_sql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test reading a standard SQL file.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr("src.sql_loader.SQL_DIR", tmp_path)
    sql_file = tmp_path / "test.sql"
    sql_file.write_text("SELECT 1;", encoding="utf-8")

    assert read_sql("test.sql") == "SELECT 1;"


def test_render_sql_template_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test rendering a SQL template successfully.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr("src.sql_loader.SQL_DIR", tmp_path)
    sql_template = tmp_path / "test.sql.j2"
    sql_template.write_text("SELECT {{ val }} FROM {{ table }};", encoding="utf-8")

    rendered = render_sql_template("test.sql.j2", val="id", table="users")
    assert rendered == "SELECT id FROM users;"


def test_render_sql_template_missing_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test rendering a SQL template with missing variables raises KeyError.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr("src.sql_loader.SQL_DIR", tmp_path)
    sql_template = tmp_path / "test.sql.j2"
    sql_template.write_text(
        "SELECT {{ val }} FROM {{ table }} WHERE {{ filter }};", encoding="utf-8"
    )

    with pytest.raises(KeyError) as exc_info:
        render_sql_template("test.sql.j2", val="id")

    assert "Missing SQL template variables for test.sql.j2: filter, table" in str(exc_info.value)


def test_render_sql_template_jinja_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test rendering a SQL template with advanced Jinja2 control structures.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr("src.sql_loader.SQL_DIR", tmp_path)
    sql_template = tmp_path / "test.sql.j2"
    template_content = (
        "SELECT id\n"
        "FROM users\n"
        "{% if filter_active %}\n"
        "WHERE role = '{{ role }}'\n"
        "{% endif %}\n"
        "ORDER BY {% for col in order_cols %}"
        "{{ col }}{% if not loop.last %}, {% endif %}"
        "{% endfor %};"
    )
    sql_template.write_text(template_content, encoding="utf-8")

    # When filter is active
    rendered = render_sql_template(
        "test.sql.j2",
        filter_active=True,
        role="admin",
        order_cols=["name", "created_at"],
    )
    expected = "SELECT id\nFROM users\n\nWHERE role = 'admin'\n\nORDER BY name, created_at;"
    assert rendered == expected
