import json
import numpy as np
import sys
import os

# Check for correct number of command-line arguments
if len(sys.argv) != 3:
    print("Usage: python convert_poses.py <input_json_file> <output_directory>")
    sys.exit(1)

# Read input file and output directory from command-line arguments
json_file = sys.argv[1]
output_dir = sys.argv[2]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Extract 3D poses
poses3d_list = [item["poses3d_world"] for item in data]

# Convert to numpy array and reshape to (-1, 17, 3)
poses3d_array = np.array(poses3d_list, dtype=np.float32).reshape(-1, 17, 3)

# Save as .npz file with the expected key name
output_path = os.path.join(output_dir, "h36m_preds.npz")
np.savez(output_path, coords3d_pred_world=poses3d_array)

print(f"Saved 3D pose data to {output_path}")
