"""Frame preprocessing shared by training (export) and serving.

Training images were produced by scripts/preprocess.py: cv2.imread (BGR) ->
cv2.resize(size, INTER_AREA) -> saved as JPEG -> loaded with PIL as RGB ->
ToTensor -> ImageNet normalisation. `to_input` reproduces that for a live frame.
"""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> BGR uint8 image. Raises ValueError on garbage input."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def resize(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Same square resize as the preprocess stage (aspect ratio is not kept)."""
    if img_bgr.shape[0] == size and img_bgr.shape[1] == size:
        return img_bgr
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)


def to_input(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR image -> normalised float32 tensor of shape (1, 3, size, size)."""
    rgb = cv2.cvtColor(resize(img_bgr, size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def brightness(img_bgr: np.ndarray, size: int = 224) -> float:
    """Mean grey level (0-255) of the resized frame: the drift signal."""
    gray = cv2.cvtColor(resize(img_bgr, size), cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
