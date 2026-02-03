import unittest
import numpy as np
from calibcam import camfunctions_ag
from calibcamlib import dist_OpenCV

class TestDistortFunctions(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)  # For reproducibility

    def test_multiple_inputs(self):
        test_shapes = [
            (1, 5, 3),
            (2, 4, 3),
            (3, 6, 3),
            (5, 5, 3),
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                boards_coords_ideal = np.random.randn(*shape).astype(np.float32)
                ks = np.random.randn(5).astype(np.float32)

                output1 = camfunctions_ag.distort(boards_coords_ideal, ks)[..., 0:2]
                output2 = dist_OpenCV.distort(boards_coords_ideal, ks, fastmath=False)

                # Check that outputs are close
                np.testing.assert_allclose(output1, output2, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
