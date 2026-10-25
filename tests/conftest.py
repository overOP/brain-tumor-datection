import io
from pathlib import Path

import pytest
from PIL import Image

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "best.pt"


@pytest.fixture
def image_bytes() -> bytes:
    """A plain 640x640 image, so the tests don't depend on the dataset."""
    img = Image.new("RGB", (640, 640), (40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()