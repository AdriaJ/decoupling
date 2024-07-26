import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
import pyxu.operator as pxop


seed = 697

ds_factor = 8

data_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/pipeline/data"

if __name__ == "__main__":
    l2s = [1e-3, 1e-2, 1e-1, 1.]

    exp_path = os.path.join(data_path, str(seed))
    save_path = os.path.join(exp_path, "reco")

    # Load the measurements
    data = np.load(os.path.join(exp_path, "data.npz"))
    img, background = data["img"], data["background"]
    measurements = data["measurements"]

    # Extract parameters
    Ngrid = img.shape[0]
    Nmeas = measurements.shape[0]
    ds_factor = Ngrid // Nmeas

    # Common operators
    # todo: maybe to be read later from config file
    kernel_std = 5  # Gaussian kernel std
    kernel_width = 3 * 2 * kernel_std + 1
    norm_meas = (np.sqrt(2 * np.pi) * kernel_std)
    kernel_measurement = np.exp(-0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / (kernel_std ** 2))
    kernel_measurement /= norm_meas

    regul_std2 = 2 * kernel_std ** 2
    norm_regul = np.sqrt(2 * np.pi * regul_std2)
    diffs = np.arange(0, 4 * np.sqrt(regul_std2), ds_factor)
    diffs = np.hstack([-diffs[1:][::-1], diffs])
    kernel_regul = np.exp(-0.5 * diffs ** 2 / regul_std2)
    kernel_regul /= norm_regul

    for i, lambda2 in enumerate(l2s):
        # usage of lambda 2
        M_kernel = kernel_regul / lambda2
        M_kernel[M_kernel.shape[0] // 2] += 1

        regul_width = kernel_regul.shape[0]
        h = np.zeros(Nmeas)
        h[:regul_width] = M_kernel
        h = np.roll(h, -regul_width // 2 + 1)
        hm1 = sfft.irfft(1 / sfft.rfft(h))
        MlambdaInv = pxop.Convolve(
            arg_shape=Nmeas,
            kernel=[hm1, ],
            center=[0, ],
            mode="wrap",  # constant
            enable_warnings=True, )
        MlambdaInv.lipschitz = MlambdaInv.estimate_lipschitz(method="svd", tol=1e-4)

        tk_res = MlambdaInv(measurements)
        tmp_tk = np.zeros(Ngrid)
        tmp_tk[ds_factor // 2::ds_factor] = tk_res
        tk_solution = np.convolve(tmp_tk, kernel_measurement, mode='same')
        np.savez(os.path.join(save_path, f"tk_lambda2_{i}"), tk_solution=tk_solution, lambda2=np.r_[lambda2])


        # Plot the figure
        plt.figure()
        plt.plot(np.arange(tk_solution.shape[0]), tk_solution)
        plt.hlines(0, 0, tk_solution.shape[0], color='black', alpha=.5)
        plt.suptitle(f"Reconstruction with lambda2 = {lambda2:.2e}")
        plt.show()
