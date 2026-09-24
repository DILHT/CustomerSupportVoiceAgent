from agent import _noise_cancellation


def test_disabled_when_flag_true(monkeypatch):
    monkeypatch.setenv("DISABLE_NOISE_CANCELLATION", "true")
    assert _noise_cancellation() is None


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("DISABLE_NOISE_CANCELLATION", raising=False)
    assert _noise_cancellation() is not None
