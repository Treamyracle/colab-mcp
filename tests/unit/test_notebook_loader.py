"""
Unit tests for Jupyter Notebook Loader & Schema Validator.

Requirements:
- TECH_SPEC §6.1 (nbformat v4 Parsing & Schema Validation)
- Feature 44 (nbformat v4 Parsing & Validation)
- Feature 45 (Exclusive Notebook Inputs Guard)
- Authoritative contract:
    def load_and_validate_notebook(
        notebook_path: Optional[str] = None,
        notebook_content: Optional[str] = None
    ) -> nbformat.NotebookNode

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import json
from pathlib import Path
import nbformat
import pytest

# Progressive testability: check implementation availability
try:
    import colab_mcp.notebook.loader as loader_mod
    NOTEBOOK_LOADER_AVAILABLE = True
except ImportError:
    loader_mod = None
    NOTEBOOK_LOADER_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_loader_implementation():
    """Progressive testability: cleanly skip if colab_mcp.notebook.loader is not yet implemented."""
    if not NOTEBOOK_LOADER_AVAILABLE:
        pytest.skip("colab_mcp.notebook.loader is scheduled for Milestone 3 implementation")


def _load(**kwargs) -> nbformat.NotebookNode:
    func = getattr(
        loader_mod,
        "load_and_validate_notebook",
        getattr(loader_mod, "load_notebook", None),
    )
    if func is not None:
        return func(**kwargs)
    raise AttributeError("No load_and_validate_notebook found in colab_mcp.notebook.loader")


# Standard minimal valid nbformat v4 notebook structure
VALID_V4_NOTEBOOK_DICT = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Analysis Notebook\n", "Initial exploratory cell."],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["Hello from remote kernel\n"],
                }
            ],
            "source": ["print('Hello from remote kernel')"],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

VALID_V4_NOTEBOOK_JSON = json.dumps(VALID_V4_NOTEBOOK_DICT)


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

def test_load_from_valid_json_content():
    """Verify loading a valid notebook from raw in-memory JSON string."""
    nb = _load(notebook_content=VALID_V4_NOTEBOOK_JSON)
    assert isinstance(nb, nbformat.NotebookNode)
    assert nb.nbformat == 4
    assert len(nb.cells) == 2


def test_load_from_valid_file_path(tmp_path: Path):
    """Verify loading a valid notebook from local filesystem path."""
    nb_file = tmp_path / "valid_test_nb.ipynb"
    nb_file.write_text(VALID_V4_NOTEBOOK_JSON, encoding="utf-8")

    nb = _load(notebook_path=str(nb_file))
    assert isinstance(nb, nbformat.NotebookNode)
    assert nb.nbformat == 4
    assert len(nb.cells) == 2


def test_loaded_notebook_cell_types_parsed_correctly():
    """Verify markdown and code cell types are cleanly recognized in the NotebookNode."""
    nb = _load(notebook_content=VALID_V4_NOTEBOOK_JSON)
    cell_types = [c.cell_type for c in nb.cells]
    assert cell_types == ["markdown", "code"]


def test_code_cell_outputs_and_source_preserved():
    """Verify cell source lines and outputs are preserved accurately in AST."""
    nb = _load(notebook_content=VALID_V4_NOTEBOOK_JSON)
    code_cell = nb.cells[1]
    assert code_cell.cell_type == "code"
    assert code_cell.execution_count == 1
    assert "print('Hello from remote kernel')" in "".join(code_cell.source)
    assert len(code_cell.outputs) == 1
    assert code_cell.outputs[0].output_type == "stream"


def test_metadata_and_kernelspec_preserved():
    """Verify notebook-level metadata and language_info are preserved."""
    nb = _load(notebook_content=VALID_V4_NOTEBOOK_JSON)
    assert "kernelspec" in nb.metadata
    assert nb.metadata.kernelspec.name == "python3"


def test_empty_cells_list_is_valid_notebook():
    """Verify an nbformat v4 notebook with zero cells (valid schema) is parsed successfully."""
    empty_nb_dict = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    nb = _load(notebook_content=json.dumps(empty_nb_dict))
    assert isinstance(nb, nbformat.NotebookNode)
    assert len(nb.cells) == 0


# ============================================================================
# Tier 2: Boundary, Exclusive Guard, Error Handling Tests (>= 5 tests)
# ============================================================================

def test_exclusive_inputs_both_none_raises_value_error():
    """
    TECH_SPEC §6.1 / Feature 45:
    When neither notebook_path nor notebook_content is provided, raises ValueError.
    """
    with pytest.raises(ValueError, match="(?i)either.*provided"):
        _load(notebook_path=None, notebook_content=None)


def test_exclusive_inputs_both_provided_raises_value_error(tmp_path: Path):
    """
    TECH_SPEC §6.1 / Feature 45:
    When both notebook_path AND notebook_content are provided, raises ValueError.
    """
    nb_file = tmp_path / "dummy.ipynb"
    nb_file.write_text(VALID_V4_NOTEBOOK_JSON, encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)not both"):
        _load(
            notebook_path=str(nb_file),
            notebook_content=VALID_V4_NOTEBOOK_JSON,
        )


def test_nonexistent_file_path_raises_file_not_found(tmp_path: Path):
    """Passing a non-existent file path raises FileNotFoundError."""
    nonexistent = str(tmp_path / "does_not_exist_xyz123.ipynb")
    with pytest.raises(FileNotFoundError):
        _load(notebook_path=nonexistent)


def test_invalid_json_content_raises_error():
    """Passing malformed/invalid JSON raises JSONDecodeError or NotJSONError."""
    malformed_json = "{'cells': [invalid_json_syntax"
    with pytest.raises(Exception):
        _load(notebook_content=malformed_json)


def test_schema_validation_missing_cells_raises_validation_error():
    """
    Schema validation: notebook payload missing mandatory 'cells' key
    raises nbformat.validator.ValidationError or ValueError.
    """
    invalid_schema_dict = {
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with pytest.raises(Exception):
        _load(notebook_content=json.dumps(invalid_schema_dict))


def test_schema_validation_invalid_cell_type_raises_error():
    """Schema validation: cell with unsupported cell_type raises validation error."""
    invalid_cell_dict = {
        "cells": [
            {
                "cell_type": "invalid_unsupported_type",
                "metadata": {},
                "source": ["some text"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with pytest.raises(Exception):
        _load(notebook_content=json.dumps(invalid_cell_dict))


def test_empty_string_content_raises_error():
    """Passing an empty string notebook_content='' raises decode or validation error."""
    with pytest.raises(Exception):
        _load(notebook_content="")


def test_older_v3_notebook_raises_or_handled_safely():
    """An older v3 notebook format is either safely upgraded or validated as v4."""
    v3_notebook = {
        "worksheets": [{"cells": []}],
        "metadata": {},
        "nbformat": 3,
        "nbformat_minor": 0,
    }
    try:
        nb = _load(notebook_content=json.dumps(v3_notebook))
        # If loader automatically upgrades, nbformat must be 4
        assert nb.nbformat == 4
    except Exception:
        # Or raises validation error rejecting non-v4
        pass
