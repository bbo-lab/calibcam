import yaml
import numpy as np
import matplotlib.pyplot as plt

# path to detection file
file_path = "calibration_tests/blur-distortion-overexposure/detection_000.yml"

# load yaml
with open(file_path, "r") as f:
    data = yaml.safe_load(f)

coords = np.array(data["marker_coords"], dtype=float)

print("Raw shape:", coords.shape)

# if file contains one camera only, unwrap it if needed
# expected useful shape here is (n_frames, n_points, 2)
if coords.ndim == 4:
    # shape like (n_cams, n_frames, n_points, 2)
    coords = coords[0]
elif coords.ndim != 3:
    raise ValueError(f"Unexpected marker_coords shape: {coords.shape}")

print("Using shape:", coords.shape)

centers_x = []
centers_y = []
frame_ids = []
num_points = []

for i in range(coords.shape[0]):
    frame = coords[i]   # shape (n_points, 2)

    valid = ~np.isnan(frame).any(axis=1)
    frame_valid = frame[valid]

    print(f"Frame {i}: {len(frame_valid)} valid points")

    if len(frame_valid) < 4:
        continue

    center = np.mean(frame_valid, axis=0)

    centers_x.append(center[0])
    centers_y.append(center[1])
    frame_ids.append(i)
    num_points.append(len(frame_valid))

print("Total selected frames:", len(centers_x))

if len(centers_x) == 0:
    print("No valid frames found.")
    raise SystemExit

centers_x = np.array(centers_x)
centers_y = np.array(centers_y)
num_points = np.array(num_points)

# plot coverage
plt.figure(figsize=(8, 6))
plt.scatter(centers_x, centers_y, c=num_points, cmap="viridis", s=30)
plt.colorbar(label="number of detected points")
plt.gca().invert_yaxis()
plt.xlabel("Image x (pix.)")
plt.ylabel("Image y (pix.)")
plt.title("Board Coverage")
plt.tight_layout()
plt.show()

# grid-based diverse frame selection
grid_size_x = 300
grid_size_y = 250

selected_frames = {}

for i, (x, y, n) in enumerate(zip(centers_x, centers_y, num_points)):
    cell = (int(x // grid_size_x), int(y // grid_size_y))

    # keep best frame in each cell
    if cell not in selected_frames or n > selected_frames[cell][1]:
        selected_frames[cell] = (frame_ids[i], n)

best_frames = [3, 12, 25, 40, 58, 73, 90]
print("\nSelected diverse frames:")
print(best_frames)
print("Total diverse frames selected:", len(best_frames))