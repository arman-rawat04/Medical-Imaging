import os
import cv2
import scipy.io
import numpy as np
import pandas as pd

# === Directory Setup for Kaggle ===
encoder_dir = "encoder_training"
decoder_dir = "decoder_training"
os.makedirs(encoder_dir, exist_ok=True)
os.makedirs(decoder_dir, exist_ok=True)

# === Utility Functions ===
def get_number_of_frames(mat_path):
    data = scipy.io.loadmat(mat_path)
    valid_keys = [key for key in data.keys() if not key.startswith('__')]
    if not valid_keys:
        raise ValueError("No valid data found in the .mat file.")
    key = valid_keys[0]
    frames = data[key]
    if frames.ndim == 3:
        return frames.shape[0], frames
    elif frames.ndim == 2:
        return 1, frames[np.newaxis, ...]
    else:
        raise ValueError(f"Unexpected data shape: {frames.shape}")

def remove_substring(main_string, substring_to_remove):
    return main_string.replace(substring_to_remove, '')

# === Read Excel File ===
df = pd.read_excel(r"C:\Users\AnirudhVijan\Downloads\OneDrive_2025-03-06\HMC-QU Dataset-Kaggle\A4C.xlsx")

# === Paths ===
video_root = r"C:\Users\AnirudhVijan\Downloads\OneDrive_2025-03-06\HMC-QU Dataset-Kaggle\HMC-QU\A4C"
mask_root = r"C:\Users\AnirudhVijan\Downloads\OneDrive_2025-03-06\HMC-QU Dataset-Kaggle\LV Ground-truth Segmentation Masks"

# === Frame Extraction Loop ===
count_total = 0
value_sum = 0
frame_id = 0  # Unique frame index for naming

for file in os.listdir(mask_root):
    mask_path = os.path.join(mask_root, file)
    try:
        mask_frame_count, mask_frames = get_number_of_frames(mask_path)
    except Exception as e:
        print(f"Skipping {file}: {e}")
        continue

    count_total += mask_frame_count

    # Get related video
    video_name_raw = remove_substring(mask_path, mask_root)
    name = video_name_raw[6:-4]  # Assuming this trims appropriately
    print("Processing:", name)

    try:
        value = df.loc[df.iloc[:, 0] == f'{name}', df.columns[7]].values[0]
    except IndexError:
        print(f"Skipping {name}: No matching entry in Excel.")
        continue

    value_sum += value
    startind = value-1
    endind = value + mask_frame_count-1

    video_path = os.path.join(video_root, name + ".avi")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, startind)

    for i in range(mask_frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {startind + i} from {video_path}")
            break

        frame_filename = f"frame_{frame_id:06d}.png"
        mask_filename = f"mask_{frame_id:06d}.png"

        # Save video frame
        cv2.imwrite(os.path.join(encoder_dir, frame_filename), frame)

        # Save corresponding mask frame
        mask = mask_frames[i]
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(decoder_dir, mask_filename), mask)

        frame_id += 1

    cap.release()
    # break

print("✅ Total mask frames processed:", count_total)
print("✅ Total value sum (start indices):", value_sum)
print("✅ Saved frames to:", encoder_dir)
print("✅ Saved masks to:", decoder_dir)
