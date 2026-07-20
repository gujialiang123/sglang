import pytest


@pytest.fixture(autouse=True)
def _enforce_runtime_context_isolation():
    from sglang.srt.runtime_context import reset_context

    reset_context()
    yield
    reset_context()
