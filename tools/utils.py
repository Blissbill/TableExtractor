from typing import Iterator, List

import cv2
import numpy as np
from fitz import fitz

from tools.models import Rectangle


def pdf_to_images(path: str) -> Iterator[np.ndarray]:
    doc = fitz.open(path)
    result = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        result.append(np.ascontiguousarray(image[..., [2, 1, 0]]))
    return result


def draw_rects(image: np.array, coordinates: List[Rectangle], color=(0, 255, 0), thickness=2):
    for coord in coordinates:
        if coord.parent_index == -1:
            cv2.rectangle(image, (coord.left, coord.top), (coord.left + coord.width, coord.top + coord.height),
                          (255, 0, 0),
                          1)
            cv2.putText(image, str(coord.index), (coord.left, coord.top + coord.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (coord.left, coord.top), (coord.left + coord.width, coord.top + coord.height), color,
                          thickness)
            cv2.putText(image, str(coord.index), (coord.left, coord.top + coord.height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, thickness, cv2.LINE_AA)