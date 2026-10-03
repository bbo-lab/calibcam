import cv2
import numpy as np
from svidreader.video_supplier import VideoSupplier

class CLAHEPreprocess(VideoSupplier):
    def __init__(self, reader):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.reader = reader

    def read(self, index, force_type=np):
        frame = self.reader.read(index=index, force_type=force_type)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Convert back to 3-channel (important for pipeline compatibility)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gray_3ch