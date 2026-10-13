import os
import time
from datetime import datetime
import cv2
import requests
from concurrent.futures import ThreadPoolExecutor


class WebcamServingClient:
    """Class responsible for managing webcam frame capture and drawing Bounding Boxes in real time."""

    def __init__(self, api_url: str, check_interval: float = 0.5):
        self.api_url = api_url
        self.check_interval = check_interval

        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        self.online_data_dir = os.path.join(self.project_root, "online_data")
        os.makedirs(self.online_data_dir, exist_ok=True)

        self.is_sending = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # [NEW] Keeps track of the latest detections to draw on screen asynchronously
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

                # [NEW] Updates the bounding box coordinates to be drawn in the main loop
                self.current_detections = data.get("detections", [])

                if data.get("has_detections"):
                    filename = f"rust_{timestamp}.png"
                    filepath = os.path.join(self.online_data_dir, filename)

                    # ALWAYS saves the raw/clean image in online_data for Active Learning
                    cv2.imwrite(filepath, frame)
                    print(
                        f"📸 [Active Learning] Corrosion detected! Saved to: online_data/{filename} ({data['count']} targets)")
            else:
                self.current_detections = []
        except Exception as e:
            print(f"⚠️ [Connection] Failed to communicate with the API: {e}")
            self.current_detections = []
        finally:
            self.is_sending = False

    def start_monitoring(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not access the webcam.")
            return

        print("\n" + "-" * 60)
        print("📹 Webcam Monitoring Started! Press 'q' to exit.")
        print("-" * 60)

        last_check_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()

                # Triggers background API verification
                if (current_time - last_check_time > self.check_interval) and not self.is_sending:
                    self.is_sending = True
                    last_check_time = current_time
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    self.executor.submit(
                        self._send_frame_to_api, frame.copy(), timestamp)

                # [NEW] DRAWS THE BOUNDING BOXES DYNAMICALLY ON THE PREVIEW SCREEN
                for det in self.current_detections:
                    bbox = det.get("bbox")  # Gets [x1, y1, x2, y2]
                    label = det.get("label")
                    conf = det.get("confidence")

                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)

                        # Draws the defect bounding box (Green)
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        # Writes the class label and confidence score text above the box
                        text = f"{label} {conf:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Renders the window with continuous video and updated boxes
                cv2.imshow(
                    "Real-Time Corrosion Inspection - MLOps Client", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=False)
            print("📹 Capture closed cleanly by the user.")


def main():
    # Correct URL pointing to your local FastAPI endpoint
    API_URL = "http://127.0.0.1:8000/predict"

    # Instantiates the client, configuring transmission sampling every 0.5 seconds
    client = WebcamServingClient(api_url=API_URL, check_interval=0.5)
    client.start_monitoring()


if __name__ == "__main__":
    main()
