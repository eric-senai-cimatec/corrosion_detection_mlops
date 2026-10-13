import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Substitua 'src.data_automation.parse_review_to_train' pelo caminho real do seu script
from src.data_automation.parse_review_to_train import main, trigger_airflow_phase2

def test_parse_and_conversion_success(tmp_path):
    review_dir = tmp_path / "review_data"
    review_dir.mkdir()
    
    train_images = tmp_path / "data" / "images" / "train"
    train_labels = tmp_path / "data" / "labels" / "train"

    # Simula um JSON legítimo exportado pelo Labelme (Retângulo de corrosão)
    fake_labelme_data = {
        "imagePath": "sample_corrosion.jpg",
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [
            {
                "shape_type": "rectangle",
                "label": "corrosion",
                "points": [[10, 10], [50, 50]]  # x1, y1, x2, y2
            }
        ]
    }
    
    # Cria os arquivos físicos na fila de curadoria temporária
    with open(review_dir / "sample_corrosion.json", "w") as f:
        json.dump(fake_labelme_data, f)
    (review_dir / "sample_corrosion.jpg").write_text("fake_binary_image")

    # Moca os caminhos e a função do Airflow CLI para isolar o teste
    with patch("src.data_automation.parse_review_to_train.REVIEW_DIR", str(review_dir)), \
         patch("src.data_automation.parse_review_to_train.TRAIN_IMAGES_DIR", str(train_images)), \
         patch("src.data_automation.parse_review_to_train.TRAIN_LABELS_DIR", str(train_labels)), \
         patch("src.data_automation.parse_review_to_train.trigger_airflow_phase2") as mock_trigger:
        
        main()
        
        # Garante que o gatilho da CLI do Airflow foi acionado após o reprocessamento com sucesso
        mock_trigger.assert_called_once()

    # ---- VALIDAÇÃO DA CONVERSÃO MATEMÁTICA ----
    txt_output_path = os.path.join(train_labels, "sample_corrosion.txt")
    assert os.path.exists(txt_output_path)
    
    with open(txt_output_path, "r") as f:
        yolo_line = f.read().strip()
        
    # Coordenadas convertidas esperadas: Centro_X=0.3, Centro_Y=0.3, Largura=0.4, Altura=0.4
    assert yolo_line == "0 0.300000 0.300000 0.400000 0.400000"
    
    # Confere se a imagem original foi movida e o JSON limpo
    assert os.path.exists(os.path.join(train_images, "sample_corrosion.jpg"))
    assert not os.path.exists(review_dir / "sample_corrosion.json")


@patch("subprocess.run")
def test_trigger_airflow_cli(mock_run):
    """Garante que a CLI local do Airflow é chamada com os argumentos corretos"""
    trigger_airflow_phase2()
    
    # Verifica se o subprocess chamou os binários da CLI de forma correta
    mock_run.assert_called_once()
    args_called = mock_run.call_args[0][0]
    assert "dags" in args_called
    assert "trigger" in args_called
    assert "corrosion_phase2_retrain" in args_called
