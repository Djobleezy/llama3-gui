import io
import pathlib
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app"))
import ocr_utils

def test_ocr_image_basic():
    img = Image.new('RGB', (200, 80), color='white')
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((10, 10), 'HELLO', font=font, fill='black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    text = ocr_utils.ocr_image(buf.getvalue())
    assert 'HELLO' in text.upper()
