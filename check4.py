import os
import re

# === Parameters ===
directory = '/home/robros-ai/dg/IL_data/dsr_block_sort_with_mask'  # Replace with your actual directory
pattern = re.compile(r'episode_(\d+)\.hdf5')

# === Step 1: Collect Episode Numbers ===
episode_numbers = []
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        episode_numbers.append(int(match.group(1)))

if not episode_numbers:
    print("No matching files found.")
    exit()

# === Step 2: Find Missing Numbers ===
episode_numbers.sort()
min_ep = min(episode_numbers)
max_ep = max(episode_numbers)
full_set = set(range(min_ep, max_ep + 1))
existing_set = set(episode_numbers)
missing = sorted(full_set - existing_set)

# === Output ===
print(f"Found {len(episode_numbers)} episodes (min={min_ep}, max={max_ep})")
if missing:
    print(f"❌ Missing episodes: {missing}")
else:
    print("✅ No episodes missing.")
