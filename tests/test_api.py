import pytest
from fastapi.testclient import TestClient

from brain_tumor.api.main import app
from conftest import WEIGHTS

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/best.pt not found")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image(client, image_bytes):
    response = client.post("/predict", files={"file": ("scan.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "scan.jpg"
    assert isinstance(body["detections"], list)


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400