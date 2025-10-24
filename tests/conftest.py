"""
Global pytest configuration for MATISSE pipeline tests.

This fixture automatically cleans up any temporary 'IterX' directories
(created during CLI or pipeline execution) after each test run, ensuring
that no residual data remains in the working directory.
"""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cleanup_iter_dirs():
    """
    Automatically remove 'Iter1' to 'Iter4' directories after each test.

    This cleanup runs even if the test fails, and ignores errors if
    directories are missing or locked.
    """
    yield  # Run the test first

    for i in range(1, 5):  # Iter1 to Iter4
        for suffix in ("", "_OIFITS"):
            iter_dir = Path(f"Iter{i}{suffix}")
            if iter_dir.exists():
                shutil.rmtree(iter_dir, ignore_errors=True)
