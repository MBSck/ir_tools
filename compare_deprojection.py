from typing import Tuple
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np


def rotate(xx, yy, pos_angle, method="counterclockwise"
           ) -> Tuple[np.ndarray, np.ndarray]:
    if method == "counterclockwise":
        xx_rot = xx*np.cos(pos_angle)-yy*np.sin(pos_angle)
        yy_rot = xx*np.sin(pos_angle)+yy*np.cos(pos_angle)
    if method == "clockwise":
        xx_rot = xx*np.cos(pos_angle)+yy*np.sin(pos_angle)
        yy_rot = -xx*np.sin(pos_angle)+yy*np.cos(pos_angle)
    return xx_rot, yy_rot


def grid(dim: int, pixel_size: float,
         pos_angle: float, axis_ratio: float) -> np.ndarray:
    pos_angle *= u.deg.to(u.rad)
    x = np.linspace(-0.5, 0.5, dim)*dim*pixel_size
    xx_rot, yy_rot = rotate(*np.meshgrid(x, x), pos_angle)
    return xx_rot, yy_rot*axis_ratio


def make_ring(rin: float, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    radius = np.hypot(xx, yy)
    radius[radius < rin] = 0
    radius[radius > (rin+0.5)] = 0
    return radius


def fourier(image: np.ndarray, pixel_size: float,) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))
    frequency_axis = np.fft.fftfreq(image.shape[0], pixel_size*u.mas.to(u.rad))
    return fft, frequency_axis


if __name__ == "__main__":
    dim, pixel_size, rin = 2048, 0.1, 5
    pos_angle, axis_ratio = 33, 0.35
    fig, axarr = plt.subplots(3, 2)
    axarr = axarr.flatten()

    xx, yy = grid(dim, pixel_size, 0, 1)
    image = make_ring(rin, xx, yy)
    fft, _ = fourier(image, pixel_size)
    axarr[0].imshow(image)
    axarr[1].imshow(np.abs(fft))

    xx_rot, yy_rot = rotate(xx, yy, pos_angle)
    yy_rot /= axis_ratio
    image = make_ring(rin, xx_rot, yy_rot)
    fft, _ = fourier(image, pixel_size)
    axarr[2].imshow(image)
    axarr[3].imshow(np.abs(fft))

    plt.show()
