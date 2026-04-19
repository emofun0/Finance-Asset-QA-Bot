def test_health_check() -> None:
    from app.api.routes.health import health_check

    response = health_check()

    assert response["status"] == "ok"
