import os
import argparse
import numpy as np
import cv2
import warnings
from PIL import Image
from utils import *


def main(config):
    files, exps = load_exposure_txt("data/" + config.path,config.ldr_num)

    # Convert to log exposure times
    log_exps = np.log(np.array(exps))

    LDR_images = read_image(files)
    height, width, channels = LDR_images.shape[1:]

    HDR_image = np.zeros((height, width, channels), dtype=np.float32)

    for c in range(channels):
        images_channel = LDR_images[:, :, :, c].copy()

        # Solve response curve
        response_curve = solve_response_curve(images_channel, log_exps, 100.0)

        # Compute radiance map (logE)
        img_rad = compute_radiance(images_channel, log_exps, response_curve)

        HDR_image[:, :, c] = np.exp(img_rad)

    filepath = os.path.join("result", config.path)
    os.makedirs(filepath,exist_ok=True)

    # Save the HDR image as hdr file
    cv2.imwrite(
        os.path.join(filepath, "output.hdr"), cv2.cvtColor(HDR_image, cv2.COLOR_RGB2BGR)
    )

    print("Radiance map reconstruction complete.")

    # Reinhard tone mapping
    tonemap = cv2.createTonemapReinhard(
        gamma=2.2, intensity=0.0, light_adapt=0.8, color_adapt=0.0
    )

    # Alternative: Mantiuk tone mapping
    # tonemap = cv2.createTonemapMantiuk(saturation=1.2, scale=0.85, gamma=1.2)

    ldr = tonemap.process(HDR_image.astype(np.float32))

    # clip to 8 bit integer
    ldr_8bit = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)

    # saturation enforcement
    hsv = cv2.cvtColor(ldr_8bit, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
    ldr_image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)

    # Alternative: white balance
    # ldr_image=adjust_white_balance(ldr_image)

    # Alternative: adjust color LAB
    # ldr_image=correct_color_lab(ldr_image)

    print("Tonemapping complete.")

    return ldr_image


if __name__ == "__main__":
    warnings.simplefilter("error")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="01")
    parser.add_argument("--save_name", type=str, default="hdr.jpg")
    parser.add_argument("--ldr_num", type=int, default=0)
    config = parser.parse_args()

    out = main(config)

    # Save the tone mapped hdr image as jpg file
    img = Image.fromarray(out)
    filepath = os.path.join("result", config.path)
    os.makedirs(filepath, exist_ok=True)
    img.save(os.path.join(filepath, config.save_name))
