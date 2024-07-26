"""
Computation of the value of lambda_1 max with respect to the value of lambda_2.
Forward operator is convolution with a Gaussian kernel, sampled on a coarse grid.
Image model is sparse Dirac impulses and background made out of Gaussian kernels.
No penalty operator, simply identity.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pyxu.operator as pxop
import pyxu.opt.solver as pxls

import scipy.fft as sfft
import scipy.signal as sig

from pyxu.opt.stop import RelError, MaxIter
from pyxu.abc import QuadraticFunc
import pyxu.util as pxu

#image model
sbgrdb = 40
Ngrid = 800
ds_factor = 8
Nmeas = Ngrid // ds_factor
sparsity = 0.3
k = 10
ongrid = True

# measurement model
kernel_std = 5  # Gaussian kernel std
kernel_width = 3 * 2 * kernel_std + 1  # Length of the Gaussian kernel
snrdb_meas = 60
norm_meas = (np.sqrt(2 * np.pi) * kernel_std)

# regularization
# lambda1_factor = 0.3

reps = 10
l2s = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100] + [0.1 * i for i in range(2, 10)]

if __name__ == "__main__":
    res = []
    for _ in range(reps):
        seed = np.random.randint(1000)
        rng = np.random.default_rng(seed=seed)
        print(f"Seed: {seed}")


        img = np.zeros((Ngrid,))
        if ongrid:
            Neff = int(.6 * Ngrid)
            # k = int(Ngrid * sparsity)
            idx = rng.choice(Neff, k, replace=False)
            indices = idx + int(.2 * Ngrid)
            img[indices] = rng.uniform(1, 10, k)

        # PSNR : 10 * np.log10(max(img)**2 / np.std(noise)**2) = 20 * log10(max(img) / std(noise))
        bg_impulses = np.zeros((Ngrid,))
        kernel_std_bg = 5 * kernel_std
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

        power_img = np.max(img)**2  # np.sum(img**2) / img.size
        target_power_noise = power_img * 10**(-sbgrdb / 10)
        sigma = np.sqrt(target_power_noise)
        # downscaling_factor = sigma / np.std(background)
        downscaling_factor = 1
        background *= downscaling_factor

        img *= 0.5

        x = img + background

        # plt.figure()
        # plt.subplot(311)
        # plt.stem(img)
        # plt.subplot(312)
        # plt.stem(background)
        # plt.subplot(313)
        # plt.stem(x)
        # plt.suptitle("Data: source image, noise, noisy image")
        # plt.show()

        # Continuous-time convolution and evaluate on the coarse grid
        kernel_measurement = np.exp(-0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / (kernel_std ** 2))
        kernel_measurement /= norm_meas

        conv_fg = np.convolve(np.pad(img, (kernel_width//2, kernel_width//2), mode='wrap'), kernel_measurement, mode='valid')

        std_meas_bg2 = kernel_std**2 + kernel_std_bg**2
        width_meas_bg = 3 * 2 * np.ceil(np.sqrt(std_meas_bg2)).astype(int) + 1
        kernel_meas_bg = (kernel_std * kernel_std_bg * np.sqrt(2 * np.pi / std_meas_bg2) *
                          np.exp(-0.5 * ((np.arange(width_meas_bg) - (width_meas_bg - 1) / 2) ** 2) / std_meas_bg2))
        kernel_meas_bg /= (norm_meas * norm_bg1d)

        conv_bg = downscaling_factor * np.convolve(np.pad(bg_impulses, (width_meas_bg//2, width_meas_bg//2), mode='wrap'),
                                                   kernel_meas_bg, mode='valid')
        # assert conv_bg.shape == background.shape
        noiseless_y = (conv_fg + conv_bg)[ds_factor//2::ds_factor]

        sigma_noise = np.linalg.norm(noiseless_y)/Nmeas * 10**(-snrdb_meas / 20)
        noise_meas = rng.normal(0, sigma_noise, noiseless_y.shape)
        y = noiseless_y + noise_meas

        # plt.figure(figsize=(12, 5))
        # plt.suptitle("Measurements on the fine grid")
        # ylim = max(conv_fg.max(), conv_bg.max())
        # plt.subplot(121)
        # plt.ylim(top=1.05 * ylim)
        # plt.title("Foreground")
        # plt.stem(conv_fg)
        # plt.subplot(122)
        # plt.ylim(top=1.05 * ylim)
        # plt.title("Background")
        # plt.stem(conv_bg)
        # plt.show()

        # plt.figure(figsize=(10, 10))
        # plt.subplot(211)
        # plt.stem(x)
        # plt.title("Original image (with background)")
        # plt.subplot(212)
        # plt.stem(y)
        # plt.title("Noisy measurements")
        # plt.show()

        #--------------------------------------------------
        #now define H
        # Dimension: Ngrid -> Nmeas
        fOp = pxop.Convolve(
            arg_shape=img.shape,
            kernel=[kernel_measurement,],
            center=[kernel_width // 2,],
            mode="wrap",  # constant
            enable_warnings=True,
        )

        fOp.lipschitz = fOp.estimate_lipschitz(method='svd', tol=1e-3)
        ss = pxop.SubSample(Ngrid, slice(ds_factor//2, Ngrid, ds_factor))
        Hop = ss * fOp

        #--------------------------------------------------

        diff_std2 = 2 * kernel_std**2
        norm_regul = np.sqrt(2 * np.pi * diff_std2)
        # xis = np.arange(ds_factor//2, Ngrid, ds_factor)
        diffs = np.arange(0, 4 * np.sqrt(diff_std2), ds_factor)
        diffs = np.hstack([-diffs[1:][::-1], diffs])
        kernel_regul = np.exp(-0.5 * diffs**2/diff_std2)
        # kernel_regul *= kernel_std**2 / ( kernel_std_regul**2 * np.sqrt(2 * np.pi * diff_std2) )
        kernel_regul /= norm_regul

        l1max = []

        for lambda2 in l2s:
            M_kernel = kernel_regul/lambda2
            M_kernel[M_kernel.shape[0]//2] += 1

            regul_width = kernel_regul.shape[0]
            h = np.zeros(Nmeas)
            h[:regul_width] = M_kernel
            h = np.roll(h, -regul_width//2 + 1)
            hm1 = sfft.irfft(1/sfft.rfft(h))

            MlambdaInv = pxop.Convolve(
                arg_shape=Nmeas,
                kernel=[hm1,],
                center=[0,],
                mode="wrap",  # constant
                enable_warnings=True,
            )

            lip_cogram = kernel_regul.sum()


            lambda1max = np.abs(Hop.adjoint(MlambdaInv(y).ravel())).max()
            # lambda1 = lambda1_factor * lambda1max

            l1max.append(lambda1max)
        res.append(l1max)

    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    for l in res:
        plt.scatter(l2s, l, marker='+')
    plt.vlines(lip_cogram, min(l), max(l), label="Lipschitz cogram")
    plt.legend()
    plt.show()


