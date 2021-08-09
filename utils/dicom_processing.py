import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from matplotlib import animation, rc
import os



def data_preparation(data):
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def jpg2array(path):
    img = Image.open(path)
    data = np.asarray(img)[:, :, 0]
    return data_preparation(data)


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    return data_preparation(data)


def plot_img(img, size=(7, 7), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def plot_imgs(imgs, cols=4, size=7, is_rgb=True, title="", cmap='gray', img_size=(500, 500)):
    rows = len(imgs) // cols + 1
    fig = plt.figure(figsize=(cols * size, rows * size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def create_animation(ims):
    fig = plt.figure(figsize=(9, 9))
    a = ims[0]
    im = plt.imshow(a)

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 24)
    return anim


def get3ScaledImage(path):
    dicom = pydicom.read_file(path)
    img = dicom.pixel_array

    r, c = img.shape
    #     img_conv = np.empty((c, r, 3), dtype=img.dtype)
    img_conv = np.empty((r, c, 3), dtype=img.dtype)
    img_conv[:, :, 2] = img_conv[:, :, 1] = img_conv[:, :, 0] = img

    ## Step 1. Convert to float to avoid overflow or underflow losses.
    img_2d = img_conv.astype(float)

    ## Step 2. Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    ## Step 3. Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)
    img_2d_scaled.reshape([img_2d_scaled.shape[0], img_2d_scaled.shape[1], 3])

    return img_2d_scaled  # , (c, r)


def get_mri_series(mri):
    series = os.listdir(mri)
    series.sort(key=lambda x: int(x[len('Image-'):-len('.dcm')]))
    return series


def cut_border(scan):
    a = 0
    while np.max(scan[a, :, :]) <= 0:
        a += 1
    b = scan.shape[0] - 1
    while np.max(scan[b, :, :]) <= 0:
        b -= 1
    c = 0
    while np.max(scan[:, c, :]) <= 0:
        c += 1
    d = scan.shape[1] - 1
    while np.max(scan[:, d, :]) <= 0:
        d -= 1
    e = 0
    while np.max(scan[:, :, e]) <= 0:
        e += 1
    f = scan.shape[2] - 1
    while np.max(scan[:, :, f]) <= 0:
        f -= 1
    return scan[a:b, c:d, e:f]

