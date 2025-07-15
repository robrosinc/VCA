import h5py
import numpy as np
import zlib
import cv2
import os

# === Parameters ===
hdf5_path = '/home/robros-ai/dg/IL_data/dsr_block_sort_with_mask/episode_362.hdf5'  # Set your specific HDF5 file
output_dir = os.path.dirname(hdf5_path)  # Save videos in the same directory, or change if needed
frame_size = (640, 480)  # (W, H)
fps = 40  # Frames per second

# === Output Path ===
filename = os.path.basename(hdf5_path)
base_name = os.path.splitext(filename)[0]
output_video_path = os.path.join(output_dir, f"{base_name}_mask_video.mp4")

print(f"🔹 Processing {hdf5_path}")

# === Read and Decompress Masks ===
decompressed_masks = []
try:
    with h5py.File(hdf5_path, 'r') as f:
        compressed_mask_dataset = f['/observations/masks/head_camera']

        for i, compressed_bytes in enumerate(compressed_mask_dataset):
            try:
                decompressed_bytes = zlib.decompress(compressed_bytes.tobytes())
                mask_array = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(frame_size[1], frame_size[0])
                decompressed_masks.append(mask_array)
            except Exception as e:
                print(f"[ERROR] Failed to decompress frame {i} in {filename}: {e}")
except Exception as e:
    print(f"[ERROR] Could not open {filename}: {e}")

# === Write video ===
if decompressed_masks:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=False)

    for mask in decompressed_masks:
        out.write(mask)

    out.release()
    print(f"✅ Video saved to: {output_video_path}")
else:
    print(f"❌ No valid masks extracted from {filename}")
