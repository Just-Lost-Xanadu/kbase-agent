def test_health(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import store

    # store.init_db() 用的是模块级 DB_PATH，先指到临时库再启动 app：
    # 否则 TestClient 触发 lifespan 会直接建表/写开发机上真实的 data/checkpoints.sqlite
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test_checkpoints.sqlite"))

    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
