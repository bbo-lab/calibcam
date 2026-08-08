import cv2
import matplotlib.pyplot as plt

video_path = "calibration_tests/blur-distortion-overexposure/CADDX000013.MP4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load frame")
    exit()

# Original for display
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Preprocessing
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Stronger CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

# Slight blur to reduce noise
gray_preprocessed = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

# Save result for inspection
cv2.imwrite("preprocessed_frame.png", gray_preprocessed)

# Show both images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(frame_rgb)
plt.title("Original Frame")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray_preprocessed, cmap="gray")
plt.title("CLAHE + Gaussian Blur")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Saved preprocessed image as preprocessed_frame.png")