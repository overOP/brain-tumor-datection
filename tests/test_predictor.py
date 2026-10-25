import pytest

from brain_tumor.inference.predictor import Predictor
from conftest import WEIGHTS

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/best.pt not found")


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_predict_returns_a_list(predictor, image_bytes):
    assert isinstance(predictor.predict(image_bytes), list)


def test_detection_format(predictor, image_bytes):
    for det in predictor.predict(image_bytes):
        assert det["class_name"] in {"glioma", "meningioma", "pituitary"}
        assert 0.0 <= det["confidence"] <= 1.0
        assert set(det["box"]) == {"x1", "y1", "x2", "y2"}


def test_invalid_image_raises(predictor):
    with pytest.raises(ValueError):
        predictor.predict(b"this is not an image")