from fastapi.testclient import TestClient

from backend.api.app import create_app


def test_ai_route_exposes_server_duration_without_changing_response(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'test.db'}", ai_delay=0,
                     static_dir=None, ai_dir=tmp_path / "ai")
    with TestClient(app) as client:
        response = client.post("/api/ai/validate", json={})

    assert response.status_code == 422
    assert float(response.headers["X-QQPK-Server-Ms"]) >= 0
