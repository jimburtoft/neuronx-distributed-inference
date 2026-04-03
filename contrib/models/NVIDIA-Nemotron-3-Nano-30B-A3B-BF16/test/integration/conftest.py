"""Pytest configuration for Nemotron integration tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (logit validation requiring CPU reference model)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (logit validation)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow to run logit validation")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
