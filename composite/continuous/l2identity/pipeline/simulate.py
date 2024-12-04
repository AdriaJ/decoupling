"""
Simulate a sparse continuous-domain foreground and a continuous-domain background.
The foreground is a sum of Dirac impulses, while the background is a sum of Gaussian kernels.
"""

import os
import argparse
import numpy as np

import scipy.signal as sig

#image model
Ngrid = 800
ds_factor = 8
Nmeas = Ngrid // ds_factor
k = 10
ongrid = True

# measurement model
kernel_std = 5  # Gaussian kernel std
kernel_width = 3 * 2 * kernel_std + 1  # Length of the Gaussian kernel
norm_meas = (np.sqrt(2 * np.pi) * kernel_std)

# background model
kernel_std_bg = 5 * kernel_std

data_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/pipeline/data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed', default=None)
    parser.add_argument('--fgbgR', type=float, default=10.)
    parser.add_argument('--snr', type=float, default=20.)
    parser.add_argument('--r12', type=float, default=1.)

    args = parser.parse_args()

    if args.seed:
        seed = int(args.seed)
    else:
        seed = np.random.randint(1000)

    snrdb_meas = args.snr
    max_intensity = args.fgbgR
    r12 = args.r12

    rng = np.random.default_rng(seed=seed)

    img = np.zeros((Ngrid,))
    if ongrid:
        Neff = int(.6 * Ngrid)
        idx = rng.choice(Neff, k, replace=False)
        indices = idx + int(.2 * Ngrid)
        img[indices] = rng.uniform(1, max_intensity, k)

    # PSNR : 10 * np.log10(max(img)**2 / np.std(noise)**2) = 20 * log10(max(img) / std(noise))
    bg_impulses = np.zeros((Ngrid,))
    if ongrid:
        kk = 10*k
        idx = rng.choice(Neff, kk, replace=False)
        indices = idx + int(.2 * Ngrid)
        bg_impulses[indices] = 1 + rng.uniform(-.5, .5, kk)
        kernel_width_bg = 3 * 2 * kernel_std_bg + 1
        kernel_bg_1d = np.exp(-0.5 * ((np.arange(kernel_width_bg) - (kernel_width_bg - 1) / 2) ** 2) / (kernel_std_bg ** 2))
        norm_bg1d = (np.sqrt(2 * np.pi) * kernel_std_bg)
        kernel_bg_1d /= norm_bg1d
        background = sig.fftconvolve(bg_impulses, kernel_bg_1d, mode='same')

    # Continuous-time convolution and evaluate on the coarse grid
    kernel_measurement = np.exp(-0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / (kernel_std ** 2))
    kernel_measurement /= norm_meas

    conv_fg = np.convolve(np.pad(img, (kernel_width//2, kernel_width//2), mode='wrap'), kernel_measurement, mode='valid')
    meas_fg = conv_fg[ds_factor//2::ds_factor]

    std_meas_bg2 = kernel_std**2 + kernel_std_bg**2
    width_meas_bg = 3 * 2 * np.ceil(np.sqrt(std_meas_bg2)).astype(int) + 1
    kernel_meas_bg = (kernel_std * kernel_std_bg * np.sqrt(2 * np.pi / std_meas_bg2) *
                      np.exp(-0.5 * ((np.arange(width_meas_bg) - (width_meas_bg - 1) / 2) ** 2) / std_meas_bg2))
    kernel_meas_bg /= (norm_meas * norm_bg1d)

    conv_bg = np.convolve(np.pad(bg_impulses, (width_meas_bg//2, width_meas_bg//2), mode='wrap'),
                                               kernel_meas_bg, mode='valid')
    meas_bg = conv_bg[ds_factor//2::ds_factor]

    factor = r12 * np.linalg.norm(meas_bg) / np.linalg.norm(meas_fg)
    img *= factor
    meas_fg *= factor

    x = img + background
    noiseless_y = (conv_fg + conv_bg)[ds_factor//2::ds_factor]

    sigma_noise = np.linalg.norm(noiseless_y)/Nmeas * 10**(-snrdb_meas / 20)
    noise_meas = rng.normal(0, sigma_noise, noiseless_y.shape)
    y = noiseless_y + noise_meas

    save_path = os.path.join(os.getcwd(), data_path, str(seed))
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(os.getcwd(), save_path))

    np.savez(os.path.join(save_path, "data.npz"),
             img=img,
             background=background,
             measurements=y,
             noise_meas=noise_meas)
