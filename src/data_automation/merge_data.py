import os
import glob
import shutil
import sys

# 1. Project roots definition
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# 2. Dynamically locates the LATEST 'predict' folder created by YOLO
BASE_DETECT_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")

# Searches for 'predict', 'predict2', 'predict3', etc.
predict_folders = glob.glob(os.path.join(BASE_DETECT_DIR, "predict*"))

if not predict_folders:
    print(f"❌ No prediction folder found at: {BASE_DETECT_DIR}")
    sys.exit()

# Sorts folders by modification date (the most recent one last)
latest_predict_dir = max(predict_folders, key=os.path.getmtime)

# 3. Defines the labels folder based on the latest execution found
PREDICT_LABELS_DIR = os.path.join(latest_predict_dir, "labels")

print(f"🔍 Latest inference folder detected: {latest_predict_dir}")
print(f"📄 Searching for labels at: {PREDICT_LABELS_DIR}")

# ---- Destinations, loops, and shutil logic ----
ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")


def merge_datasets():
    print("Starting the merging process of the new pseudo-labeled data...")

    if not os.path.exists(PREDICT_LABELS_DIR):
        print(
            f"❌ Generated labels folder not found: {PREDICT_LABELS_DIR}")
        return

    # Ensures that destination folders exist
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

    copied_counter = 0

    # Scans the generated annotations
    for txt_file in os.listdir(PREDICT_LABELS_DIR):
        if txt_file.endswith(".txt"):
            base_name = os.path.splitext(txt_file)[0]

            # Locates the corresponding image in the online_data folder (testing common extensions)
            img_file = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                potential_img = base_name + ext
                if os.path.exists(os.path.join(ONLINE_DATA_DIR, potential_img)):
                    img_file = potential_img
                    break

            if img_file:
                # 1. Copies the .txt file to the official training labels folder
                shutil.copy2(
                    os.path.join(PREDICT_LABELS_DIR, txt_file),
                    os.path.join(TRAIN_LABELS_DIR, txt_file)
                )

                # 2. Copies the original image to the official training images folder
                shutil.copy2(
                    os.path.join(ONLINE_DATA_DIR, img_file),
                    os.path.join(TRAIN_IMAGES_DIR, img_file)
                )

                # 3. Removes the processed image from the online_data folder (Environment cleanup)
                os.remove(os.path.join(ONLINE_DATA_DIR, img_file))

                copied_counter += 1

    # Removes residual files with no detections (e.g., rust8.png) to avoid garbage accumulation
    for residual_file in os.listdir(ONLINE_DATA_DIR):
        if residual_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            os.remove(os.path.join(ONLINE_DATA_DIR, residual_file))

    print(
        f"✅ Success! {copied_counter} new images and labels merged into the official training dataset.")
    print("🧹 'online_data' folder cleaned and ready to receive new frames.")


if __name__ == "__main__":
    merge_datasets()
