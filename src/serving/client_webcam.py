import os
import time
from datetime import datetime
import cv2
import requests
from concurrent.futures import ThreadPoolExecutor


class WebcamServingClient:
    """Classe responsável por gerenciar a captura de frames da webcam e o envio assíncrono para a API."""

    def __init__(self, api_url: str, check_interval: float = 0.5):
        self.api_url = api_url
        self.check_interval = check_interval

        # Resolução da estrutura de caminhos locais
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        self.online_data_dir = os.path.join(self.project_root, "online_data")
        os.makedirs(self.online_data_dir, exist_ok=True)

        # Controle de estado encapsulado (Evita variáveis globais)
        self.is_sending = False

        # Gerenciamento seguro de Threads para processamento em background (CPU-safe)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _send_frame_to_api(self, frame, timestamp: str):
        """Envia o frame via HTTP POST e salva o arquivo em online_data se houver detecção positiva."""
        try:
            # Codifica o frame em formato .jpg diretamente em memória
            success, img_encoded = cv2.imencode(".jpg", frame)
            if not success:
                print("⚠️ [Processamento] Falha ao codificar frame para envio.")
                return

            files = {
                "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

            # Executa a requisição síncrona dentro da Thread separada
            response = requests.post(self.api_url, files=files, timeout=5)

            if response.status_code == 200:
                data = response.json()

                # Se o modelo @champion detectar corrosão (Confiança > 0.30 configurada na API)
                if data.get("has_detections"):
                    filename = f"rust_{timestamp}.png"
                    filepath = os.path.join(self.online_data_dir, filename)

                    # Escreve o frame cru no disco para re-alimentar o Active Learning
                    cv2.imwrite(filepath, frame)
                    print(
                        f"📸 [Active Learning] Corrosão localizada! Salvo em: online_data/{filename} ({data['count']} focos)")
            else:
                print(
                    f"⚠️ [API] Servidor respondeu com código de erro: {response.status_code}")

        except Exception as e:
            print(f"⚠️ [Conexão] Falha ao se comunicar com a API: {e}")
        finally:
            # Garante que a trava do estado seja liberada mesmo se a requisição estourar timeout
            self.is_sending = False

    def start_monitoring(self):
        """Inicializa o loop de captura de vídeo fluido (Thread Principal)."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Erro: Não foi possível acessar a webcam do dispositivo.")
            return

        print("\n" + "-" * 60)
        print("📹 Monitoramento da Webcam Iniciado! Pressione 'q' para fechar.")
        print("-" * 60)

        last_check_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erro: Falha ao receber o stream de vídeo da webcam.")
                    break

                current_time = time.time()

                # Gerencia o disparo assíncrono para o pool de threads
                if (current_time - last_check_time > self.check_interval) and not self.is_sending:
                    self.is_sending = True
                    last_check_time = current_time
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    # Envia uma cópia isolada do frame para evitar mutação de memória na Thread secundária
                    self.executor.submit(
                        self._send_frame_to_api, frame.copy(), timestamp)

                # Renderiza o vídeo continuamente a 30 FPS na tela principal de forma fluida
                cv2.imshow(
                    "Inspecao de Corrosao em Tempo Real - MLOps Client", frame)

                # Escuta tecla de saída
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Garante o fechamento limpo de recursos de hardware e encerramento do pool de threads
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=False)
            print("📹 Captura e conexões encerradas de forma limpa pelo usuário.")


def main():
    # URL correta apontando para o endpoint da sua API FastAPI local
    API_URL = "http://127.0.0"

    # Instancia o cliente configurando amostragem de envio a cada 0.5 segundos
    client = WebcamServingClient(api_url=API_URL, check_interval=0.5)
    client.start_monitoring()


if __name__ == "__main__":
    main()
