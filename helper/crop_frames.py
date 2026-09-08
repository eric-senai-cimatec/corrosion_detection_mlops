import os
import cv2

# Caminho da pasta com os frames
folder_path = "corrosion_best_frames_croped"

# Listar todos os arquivos de imagem
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[
    1].lower() in image_extensions]

for image_name in images:
    image_path = os.path.join(folder_path, image_name)

    # Ler a imagem
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Erro ao ler: {image_name}")
        continue

    height, width = frame.shape[:2]

    # Calcular coordenadas dos 4 quadrantes
    mid_height = height // 2
    mid_width = width // 2

    # Quadrante 1 (topo-esquerdo)
    crop1 = frame[0:mid_height, 0:mid_width]
    # Quadrante 2 (topo-direito)
    crop2 = frame[0:mid_height, mid_width:width]
    # Quadrante 3 (baixo-esquerdo)
    crop3 = frame[mid_height:height, 0:mid_width]
    # Quadrante 4 (baixo-direito)
    crop4 = frame[mid_height:height, mid_width:width]

    # Gerar nomes para os 4 pedaços
    base_name = os.path.splitext(image_name)[0]
    extension = os.path.splitext(image_name)[1]

    crops = [crop1, crop2, crop3, crop4]
    for i, crop in enumerate(crops, 1):
        crop_name = f"{base_name}_crop{i}{extension}"
        crop_path = os.path.join(folder_path, crop_name)
        cv2.imwrite(crop_path, crop)
        print(f"Salvo: {crop_name}")

    # Deletar frame original
    os.remove(image_path)
    print(f"Deletado: {image_name}")

print("Processo concluído!")
