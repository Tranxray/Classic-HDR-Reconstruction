import os
import cv2
import math
import random
import numpy as np

def convert_frac_to_float(exp:str)->float:
    if exp[-1]=='s':
        exp=exp[:-1]

    if '/' in exp:
        (t1,t2)=exp.split('/')
        time1=float(t1)
        time2=float(t2)

        return time1/time2
    else:
        time1=float(exp)
        return time1


def load_exposure_txt(path_test,ldr_num):
    filenames = []
    exposure_times = []
    with open(os.path.join(path_test, "exposures.txt")) as f:
        for line in f:
            filename, exposure = line.split()
            filenames.append(os.path.join(path_test, filename + ".jpg"))
            exposure_times.append(convert_frac_to_float(exposure))

    if ldr_num<=0: ldr_num=len(filenames)
    random_idxs = np.random.choice(range(len(filenames)), size=ldr_num, replace=False)

    chosen_filenames=[]
    chosen_exposure=[]
    for idx in random_idxs:
        chosen_filenames.append(filenames[idx])
        chosen_exposure.append(exposure_times[idx])

    print("readed exposure time: ",chosen_exposure)

    return chosen_filenames, chosen_exposure


def read_image(path):
    shape = cv2.imread(path[0]).shape

    images = np.zeros((len(path), shape[0], shape[1], shape[2]))

    print("readed images: ")

    for idx,i in enumerate(path):
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(i)
        images[idx, :, :, :] = image
    return images


def weight(z):
    return np.minimum(z, 255 - z)
    # Alternative: Guassian weight
    # return np.exp(-((z - 128) ** 2) / (2 * 30**2))


def solve_response_curve(images, log_exps, l_smooth=100.0, n_samples=100):
    """
    Solves for the camera response function g and radiance values E.
    Args:
        images: (N, H, W), N LDR images, grayscale channel
        log_exps: list of log exposure times
        l_smooth: lambda for smoothness constraint
        n_samples: number of pixel locations to sample
    Returns:
        g: camera response curve, shape (256,)
    """
    N = images.shape[0]

    height, width = images.shape[1:3]
    np.random.seed(42)
    sample_indices = [
        (np.random.randint(0, height), np.random.randint(0, width))
        for _ in range(n_samples)
    ]

    A_rows = n_samples * N + 256 + 1
    A_cols = 256 + n_samples
    A = np.zeros((A_rows, A_cols), dtype=np.float64)
    b = np.zeros((A_rows, 1), dtype=np.float64)

    k = 0
    for i, (x, y) in enumerate(sample_indices):
        for j in range(N):
            z = int(images[j, x, y])
            wij = weight(z)
            A[k, z] = wij
            A[k, 256 + i] = -wij
            b[k, 0] = wij * log_exps[j]
            k += 1

    # Smoothness term
    for z in range(1, 255):
        w_z = weight(z)
        A[k, z - 1] = l_smooth * w_z
        A[k, z] = -2 * l_smooth * w_z
        A[k, z + 1] = l_smooth * w_z
        k += 1

    # Fix g(128) = 0
    A[k, 128] = 1.0
    b[k, 0] = 0.0

    # Solve Ax = b
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:256].flatten()

    return g


def compute_radiance(images, exps, curve):
    lin_img = np.zeros(images.shape[1:], dtype=np.float64)

    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            g = np.array([curve[int(images[k][i, j])] for k in range(images.shape[0])])
            w = np.array(
                [
                    min(images[k][i, j], 255 - images[k][i, j])
                    for k in range(images.shape[0])
                ]
            )
            sumW = np.sum(w)
            if sumW > 0:
                lin_img[i, j] = np.sum(w * (g - exps) / sumW)
            else:
                lin_img[i, j] = g[images.shape[0] // 2] - exps[images.shape[0] // 2]
    return lin_img


def adjust_white_balance(img, blue_gain=0.95, red_gain=1.05):
    img = img.astype(np.float32)
    img[:, :, 0] *= blue_gain
    img[:, :, 2] *= red_gain
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def correct_color_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    a = cv2.add(a, 5) 
    b = cv2.subtract(b, 10) 

    lab_corrected = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
