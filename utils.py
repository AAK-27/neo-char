import cv2
import numpy as np

def prepare_sprite(sprite_path:str, width:int, height:int):
    """ Scales and centers the sprite on a green background """
    sprite_bgra = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    if sprite_bgra is None:
        raise ValueError(f"Could not load sprite from {sprite_path}")

    h, w = sprite_bgra.shape[:2]
    channels = sprite_bgra.shape[2]

    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    sprite_bgra = cv2.resize(sprite_bgra, (new_w, new_h))

    green = np.zeros((height, width, 3), dtype=np.uint8)
    green[:, :, 1] = 255

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    if channels == 4:
        # If the sprite has an alpha channel, all transparent pixels must be set to green
        alpha = sprite_bgra[:, :, 3:] / 255.0
        bgr = sprite_bgra[:, :, :3]
        green[y:y+new_h, x:x+new_w] = (bgr * alpha + green[y:y+new_h, x:x+new_w] * (1 - alpha)).astype(np.uint8)
    else:
        # Or else just use the og image
        green[y:y+new_h, x:x+new_w] = sprite_bgra

    return cv2.cvtColor(green, cv2.COLOR_BGR2RGB)