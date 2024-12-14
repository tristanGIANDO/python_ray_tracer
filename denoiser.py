import cv2
import numpy as np


def denoise(image: np.ndarray) -> np.ndarray:
    image_uint8 = (image * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(image_uint8, None, 10, 10, 7, 21)
    return denoised.astype(np.float32) / 255
