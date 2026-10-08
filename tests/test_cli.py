from __future__ import annotations

import sys

from agrovision import cli


def test_demo_runs(capsys):
    cli.demo()
    out = capsys.readouterr().out
    assert "AgroVision demo" in out
    assert "verdict:" in out
    # Each class verdict is shown, including the rust -> fungicide recommendation.
    assert "healthy" in out
    assert "fungicide" in out
    # The clearly-healthy leaf is handled by the deterministic rule.
    assert "via rule" in out
    assert "classified correctly" in out


def test_main_demo_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["agrovision", "demo"])
    assert cli.main() == 0
    assert "AgroVision demo" in capsys.readouterr().out


def test_main_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["agrovision"])
    assert cli.main() == 1
    assert "usage" in capsys.readouterr().out
