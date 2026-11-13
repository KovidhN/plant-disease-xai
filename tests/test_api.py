from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_predict_mock():
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["is_mock"] == True
    assert "predicted_class" in result
