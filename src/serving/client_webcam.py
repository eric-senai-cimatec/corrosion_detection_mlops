import os
import time
from datetime import datetime
import cv2
import requests
from concurrent.futures import ThreadPoolExecutor


class WebcamServingClient:
    """Classe responsável por gerenciar a captura de frames da webcam e o desenho de Bounding Boxes em tempo real."""

    def __init__(self, api_url: str, check_interval: float = 0.5):
        self.api_url = api_url
        self.check_interval = check_interval

        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        self.online_data_dir = os.path.join(self.project_root, "online_data")
        os.makedirs(self.online_data_dir, exist_ok=True)

        self.is_sending = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # [NOVO] Mantém o histórico das últimas detecções para desenhar na tela de forma assíncrona
        self.current_detections = []

    def _send_frame_to_api(self, frame, timestamp: str):
        try:
            success, img_encoded = cv2.imencode(".jpg", frame)
            if not success:
                return

            files = {
                "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            response = requests.post(self.api_url, files=files, timeout=5)

            if response.status_code == 200:
                data = response.json()

                # [NOVO] Atualiza as coordenadas das caixas que serão desenhadas no loop principal
                self.current_detections = data.get("detections", [])

                if data.get("has_detections"):
                    filename = f"rust_{timestamp}.png"
                    filepath = os.path.join(self.online_data_dir, filename)

                    # Salva SEMPRE a imagem crua/limpa no online_data para o Active Learning
                    cv2.imwrite(filepath, frame)
                    print(
                        f"📸 [Active Learning] Corrosão localizada! Salvo em: online_data/{filename} ({data['count']} focos)")
            else:
                self.current_detections = []
        except Exception as e:
            print(f"⚠️ [Conexão] Falha ao se comunicar com a API: {e}")
            self.current_detections = []
        finally:
            self.is_sending = False

    def start_monitoring(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Erro: Não foi possível acessar a webcam.")
            return

        print("\n" + "-" * 60)
        print("📹 Monitoramento da Webcam Iniciado! Pressione 'q' para fechar.")
        print("-" * 60)

        last_check_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()

                # Dispara a verificação em background
                if (current_time - last_check_time > self.check_interval) and not self.is_sending:
                    self.is_sending = True
                    last_check_time = current_time
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    self.executor.submit(
                        self._send_frame_to_api, frame.copy(), timestamp)

                # [NOVO] DESENHA AS BOUNDING BOXES DINAMICAMENTE NA TELA DE VISUALIZAÇÃO
                for det in self.current_detections:
                    bbox = det.get("bbox")  # Pega [x1, y1, x2, y2]
                    label = det.get("label")
                    conf = det.get("confidence")

                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)

                        # Desenha o retângulo da falha (Verde)
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        # Escreve o texto com a classe e a confiança em cima da caixa
                        text = f"{label} {conf:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Renderiza a janela com o vídeo contínuo e as caixas atualizadas
                cv2.imshow(
                    "Inspeção de Corrosão em Tempo Real - MLOps Client", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=False)
            print("📹 Captura encerrada de forma limpa pelo usuário.")


def main():
    # URL correta apontando para o endpoint da sua API FastAPI local
    API_URL = "http://127.0.0.1:8000/predict"

    # Instancia o cliente configurando amostragem de envio a cada 0.5 segundos
    client = WebcamServingClient(api_url=API_URL, check_interval=0.5)
    client.start_monitoring()


if __name__ == "__main__":
    main()
