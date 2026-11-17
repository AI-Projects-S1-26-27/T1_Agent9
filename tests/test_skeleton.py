"""Project-structure checks that keep the repository honest as it grows.

Owner: Jainam Parekh Naresh, Reviewer: Vaibhav Malik

Why these tests exist:
- The instructor docked a mark because accountability was unclear. Every
  source and test file must name one owner and one reviewer, and this test
  fails the build the moment a file is added without them.
- Every analysis reads its numbers from config/*.yaml. These tests only check
  that each file parses to a mapping with a schema_version; value validation
  (types, ranges, NaN, shapes) is P1 work and lives with the config loader.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "resource_war"
CONFIG = ROOT / "config"

OWNER_LINE = re.compile(r"^Owner: [^,\n]+, Reviewer: [^\n]+$", re.MULTILINE)

PACKAGES = [
    "resource_war",
    "resource_war.engine",
    "resource_war.encounter",
    "resource_war.equilibrium",
    "resource_war.battle",
    "resource_war.agents",
    "resource_war.tournament",
    "resource_war.demo",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)


def _python_files() -> list[Path]:
    return sorted([*SRC.rglob("*.py"), *(ROOT / "tests").rglob("*.py")])


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_module_names_owner_and_reviewer(path: Path) -> None:
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert docstring, f"{path.name} has no module docstring"
    assert OWNER_LINE.search(docstring), (
        f"{path.name}: module docstring needs a line 'Owner: <name>, Reviewer: <name>'"
    )


@pytest.mark.parametrize("path", sorted(CONFIG.glob("*.yaml")), ids=lambda p: p.name)
def test_config_files_parse(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} must be a YAML mapping"
    assert data.get("schema_version") == 1, f"{path.name} must declare schema_version: 1"


def test_expected_config_files_exist() -> None:
    names = {p.name for p in CONFIG.glob("*.yaml")}
    assert {"payoffs.yaml", "game.yaml", "battle.yaml", "agents.yaml", "experiments.yaml"} <= names
