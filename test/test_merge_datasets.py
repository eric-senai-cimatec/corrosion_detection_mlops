import os
import pytest
from unittest.mock import patch

# Aponta para o nome real do script do projeto (src.data_automation.merge)
from src.data_automation.merge_data import merge_datasets

def test_merge_datasets_success(tmp_path):
    online_data = tmp_path / "online_data"
    online_data.mkdir()
    
    predict_dir = tmp_path / "runs" / "detect" / "predict"
    labels_dir = predict_dir / "labels"
    labels_dir.mkdir(parents=True)
    
    train_images = tmp_path / "data" / "images" / "train"
    train_labels = tmp_path / "data" / "labels" / "train"

    (online_data / "image_detected.jpg").write_text("fake_image_data")
    (labels_dir / "image_detected.txt").write_text("0 0.5 0.5 0.2 0.2")
    (online_data / "image_residual.png").write_text("fake_residual_data")

    # Moca as constantes de destino dentro do modulo real 'merge'
    with patch("src.data_automation.merge_data.ONLINE_DATA_DIR", str(online_data)), \
         patch("src.data_automation.merge_data.TRAIN_IMAGES_DIR", str(train_images)), \
         patch("src.data_automation.merge_data.TRAIN_LABELS_DIR", str(train_labels)), \
         patch("glob.glob", return_value=[str(predict_dir)]):
        
        merge_datasets()

    assert os.path.exists(os.path.join(train_labels, "image_detected.txt"))
    assert os.path.exists(os.path.join(train_images, "image_detected.jpg"))
    assert not os.path.exists(os.path.join(online_data, "image_detected.jpg"))
    assert not os.path.exists(os.path.join(online_data, "image_residual.png"))
