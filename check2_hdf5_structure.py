import h5py
import numpy as np
import os

def check_nonzero(ds):
    """Returns True if dataset contains any non-zero value."""
    try:
        return np.count_nonzero(ds[...]) > 0
    except Exception as e:
        return f"Error reading dataset: {e}"

def print_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            shape = obj.shape
            dtype = obj.dtype
            nonzero = check_nonzero(obj)
            print(f"[DATASET] {name}")
            print(f"  ├─ Shape : {shape}")
            print(f"  ├─ DType : {dtype}")
            print(f"  └─ Non-Zero? : {nonzero}\n")
            
            if dtype.kind in ('O', 'V'):  # Object or vlen
                try:
                    sample = obj[0]  # Read the first item
                    print(f"     ├─ Sample Type  : {type(sample)}")
                    if isinstance(sample, np.ndarray):
                        print(f"     ├─ Sample Shape : {sample.shape}")
                        print(f"     └─ Sample DType : {sample.dtype}")
                    else:
                        print(f"     └─ Sample Value : {sample}")
                except Exception as e:
                    print(f"     └─ Error reading sample: {e}")
            print()
        except Exception as e:
            print(f"[ERROR] {name} — {e}")

# === Set path to your file
hdf5_path = '/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/test/episode_2.h5'
#print("Exists?", os.path.exists(hdf5_path))
with h5py.File(hdf5_path, 'r') as f:
    print(f"\nScanning HDF5 file: {hdf5_path}\n{'='*50}")
    f.visititems(print_datasets)