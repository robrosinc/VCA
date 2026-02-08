import h5py
import numpy as np
from pathlib import Path


def numeric_to_tuple(vec):
    vec = np.asarray(vec)

    if vec.shape != (14,):
        raise ValueError(f"numeric must have shape (14,), got {vec.shape}")

    d1 = d2 = d3 = 0

    # first digit
    if vec[0] == 1:
        d1 = 1
    elif vec[1] == 1:
        d1 = 2

    # second digit
    for i in range(2, 8):
        if vec[i] == 1:
            d2 = i - 2
            break

    # third digit
    for i in range(8, 14):
        if vec[i] == 1:
            d3 = i - 8
            break

    return [d1, d2, d3]


def convert_dataset(ds):
    data = ds[()]  # shape (T, 14)

    if data.ndim != 2 or data.shape[1] != 14:
        raise ValueError(f"Expected (T, 14), got {data.shape}")

    converted = np.array([numeric_to_tuple(row) for row in data], dtype=np.int64)
    return converted


def process_hdf5_file(path):
    with h5py.File(path, "r+") as f:
        for key in ["/prompts/numeric", "/prompts/numeric2"]:
            if key in f:
                ds = f[key]
                print(f"Processing {path.name}:{key} {ds.shape}")

                converted = convert_dataset(ds)

                # delete old dataset and replace
                del f[key]
                f.create_dataset(key, data=converted, compression="gzip")

            else:
                print(f"Skipping {path.name}:{key} (not found)")


def main(hdf5_dir):
    hdf5_dir = Path(hdf5_dir)

    for path in hdf5_dir.glob("*.h5"):
        process_hdf5_file(path)

    for path in hdf5_dir.glob("*.hdf5"):
        process_hdf5_file(path)


if __name__ == "__main__":
    main("/root/dataset")

