import os
import pytest
from unittest.mock import patch, MagicMock

mock_client = MagicMock()
mock_version = MagicMock()
mock_version.source = "file:///home/eric/fake_champion/best.pt"
mock_client.get_model_version_by_alias.return_value = mock_version

mock_yolo_instance = MagicMock()
mock_box = MagicMock()

# CORREÇÃO CRÍTICA: Simula o retorno de uma lista de floats puros (x1, y1, x2, y2)
# compatíveis com o serializador JSON nativo do Python
mock_tensor_xyxy = MagicMock()
mock_tensor_xyxy.tolist.return_value = [10.0, 20.0, 50.0, 60.0]
mock_box.xyxy = [mock_tensor_xyxy]  # Bate com o índice [0] mapeado no script original

# Simula o ID de classe e o item numérico do PyTorch de forma limpa
mock_cls_tensor = MagicMock()
mock_cls_tensor.item.return_value = 0
mock_box.cls = [mock_cls_tensor]

mock_results = MagicMock()
mock_results.boxes = [mock_box]
mock_results.names = {0: "corrosion"}
mock_yolo_instance.predict.return_value = [mock_results]

@patch("src.data_automation.send_to_review.MlflowClient", return_value=mock_client)
@patch("src.data_automation.send_to_review.YOLO", return_value=mock_yolo_instance)
@patch("cv2.imread")
def test_send_to_review_generation(mock_imread, mock_yolo, mock_mlflow, tmp_path):
    online_dir = tmp_path / "online_data"
    online_dir.mkdir()
    review_dir = tmp_path / "review_data"
    review_dir.mkdir()
    
    (online_dir / "frame_01.jpg").write_text("fake_pixels")
    mock_imread.return_value = MagicMock(shape=(100, 100, 3))

    with patch("src.data_automation.send_to_review.ONLINE_DATA_DIR", str(online_dir)), \
         patch("src.data_automation.send_to_review.REVIEW_DIR", str(review_dir)):
        
        from src.data_automation.send_to_review import run_active_learning_inference
        run_active_learning_inference()
        
        assert os.path.exists(review_dir / "frame_01.jpg")
        assert os.path.exists(review_dir / "frame_01.json")
