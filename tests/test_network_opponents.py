import json
from pathlib import Path

from scripts.experiments import network_opponents


def test_exploit_cache_tracks_model_spots_and_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-v1")
    source = tmp_path / "backend/policy/solver.py"
    source.parent.mkdir(parents=True)
    source.write_text("solver-v1")
    monkeypatch.setattr(network_opponents, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(network_opponents, "CACHE_SOURCE_DIRS", ("backend/policy",))
    legacy = tmp_path / "out/policy/arch-demo/exploitability.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"summary": {"legacy": True}}))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--out") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": {"run": len(calls)}}))

    monkeypatch.setattr(network_opponents.subprocess, "run", fake_run)
    assert network_opponents.exploit("demo", model, 40) == {"run": 1}
    assert network_opponents.exploit("demo", model, 40) == {"run": 1}
    assert network_opponents.exploit("demo", model, 60) == {"run": 2}
    model.write_bytes(b"model-v2")
    assert network_opponents.exploit("demo", model, 60) == {"run": 3}
    source.write_text("solver-v2")
    assert network_opponents.exploit("demo", model, 60) == {"run": 4}
    assert len(calls) == 4
    assert json.loads(legacy.read_text()) == {"summary": {"legacy": True}}
