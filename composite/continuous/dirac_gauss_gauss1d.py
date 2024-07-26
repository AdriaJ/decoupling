"""
Forward operator is convolution with a Gaussian kernel.
Image model is sparse Dirac impulses and additive gaussian noise.
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
seed = 1
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
kernel_std_regul = kernel_std / 2
norm_regul = (np.sqrt(2 * np.pi) * kernel_std_regul)
lambda1_factor = 0.15
lambda2 = 1e-4
eps = 1e-5

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    rng = np.random.default_rng(seed=seed)


    img = np.zeros((Ngrid,))
    if ongrid:
        Neff = int(.6 * Ngrid)
        # k = int(Ngrid * sparsity)
        idx = rng.choice(Neff, k, replace=False)
        indices = idx + int(.2 * Ngrid)
        img[indices] = rng.uniform(1, 10, k)

    # PSNR : 10 * np.log10(max(img)**2 / np.std(noise)**2) = 20 * log10(max(img) / std(noise))
    bg_impulses = np.zeros((Ngrid,))
    kernel_std_bg = np.sqrt(.5) * kernel_std
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

    x = img + background

    plt.figure()
    plt.subplot(311)
    plt.stem(img)
    plt.subplot(312)
    plt.stem(background)
    plt.subplot(313)
    plt.stem(x)
    plt.suptitle("Data: source image, noise, noisy image")
    plt.show()

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

    plt.figure(figsize=(12, 5))
    plt.suptitle("Measurements on the fine grid")
    ylim = max(conv_fg.max(), conv_bg.max())
    plt.subplot(121)
    plt.ylim(top=1.05 * ylim)
    plt.title("Foreground")
    plt.stem(conv_fg)
    plt.subplot(122)
    plt.ylim(top=1.05 * ylim)
    plt.title("Background")
    plt.stem(conv_bg)
    plt.show()

    # n_kernel_measurement = kernel_measurement/kernel_measurement.sum()
    # n_kernel_bg_1d = kernel_bg_1d/kernel_bg_1d.sum()
    # np.convolve(n_kernel_measurement, n_kernel_bg_1d).max()


    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.stem(x)
    plt.title("Original image (with background)")
    plt.subplot(212)
    plt.stem(y)
    plt.title("Noisy measurements")
    plt.show()

    #--------------------------------------------------

    diff_std2 = 2 * (kernel_std**2 - kernel_std_regul**2)
    # xis = np.arange(ds_factor//2, Ngrid, ds_factor)
    diffs = np.arange(0, 4 * np.sqrt(diff_std2), ds_factor)
    diffs = np.hstack([-diffs[1:][::-1], diffs])
    kernel_regul = np.exp(-0.5 * diffs**2/diff_std2)
    kernel_regul *= kernel_std**2 / ( kernel_std_regul**2 * np.sqrt(2 * np.pi * diff_std2) )
    kernel_regul /= norm_regul
    M_kernel = kernel_regul/lambda2
    M_kernel[M_kernel.shape[0]//2] += 1

    regul_width = kernel_regul.shape[0]
    h = np.zeros(Nmeas)
    h[:regul_width] = M_kernel
    h = np.roll(h, -regul_width//2 + 1)
    hm1 = sfft.irfft(1/sfft.rfft(h))

    Mlambda = pxop.Convolve(
        arg_shape=Nmeas,
        kernel=[M_kernel,],
        center=[M_kernel.shape[0]//2,],
        mode="constant",
        enable_warnings=True,
    )
    # mat = Mlambda.asarray()
    # mat1 = np.linalg.inv(mat)

    MlambdaInv = pxop.Convolve(
        arg_shape=Nmeas,
        kernel=[hm1,],
        center=[0,],
        mode="wrap",  # constant
        enable_warnings=True,
    )

    # import pyxu.abc as pxa
    # MlambdaInv = pxa.LinOp.from_array(mat1)
    MlambdaInv.lipschitz = MlambdaInv.estimate_lipschitz(method="svd", tol=1e-4)


    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(mat)
    # plt.subplot(122)
    # plt.imshow(mat1)
    # plt.show()
    #
    # mat2 = MlambdaInv.asarray()
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(np.linalg.inv(mat))
    # plt.subplot(122)
    # plt.imshow(mat2)
    # plt.show()

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

    lambda1max = np.abs(Hop.adjoint(MlambdaInv(y).ravel())).max()
    lambda1 = lambda1_factor * lambda1max

    loss = QuadraticFunc((1, Nmeas), Q=MlambdaInv).asloss(y.ravel()) * Hop
    # loss.diff_lipschitz = loss.estimate_diff_lipschitz(method='svd')  # Mlambda.lipschitz  # fOp.lipschitz = 1.

    regul = lambda1 * pxop.PositiveL1Norm(Ngrid)

    stop_crit = RelError(eps=eps, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

    print("Decoupled solving...")
    pgd = pxls.PGD(loss, g=regul, show_progress=False)
    start = time.time()
    pgd.fit(x0=np.zeros(img.size), stop_crit=stop_crit)
    pgd_time = time.time() - start

    _, hist = pgd.stats()
    x1 = pgd.solution()

    Mresiduals = MlambdaInv(y - Hop(x1))
    varphi_std2 = kernel_std**2 - 2 * kernel_std_regul**2
    xis = np.arange(0, 3 * np.sqrt(varphi_std2) + 1)
    varphi_sup = np.hstack([-xis[1:][::-1], xis])
    varphi_kernel = np.exp(-0.5 * varphi_sup**2 / varphi_std2)
    varphi_kernel *= (kernel_std/(2 * np.pi * (kernel_std_regul**2) * np.sqrt(varphi_std2)))
    tmp = np.zeros(Ngrid)
    tmp[ds_factor//2::ds_factor] = Mresiduals
    x2 = np.convolve(tmp, varphi_kernel, mode='same') / lambda2

    plt.figure(figsize=(12, 11))
    ylim = max(img.max(), x1.max())
    plt.subplot(421)
    plt.ylim(top=1.05 * ylim)
    plt.stem(img)
    plt.title("Source foreground")
    plt.subplot(422)
    plt.ylim(top=1.05 * ylim)
    plt.stem(x1)
    plt.title("Recovered foreground")

    ylim = max(background.max(), x2.max())
    plt.subplot(423)
    plt.ylim(top=1.05 * ylim)
    # plt.stem(background)
    plt.scatter(np.arange(Ngrid), background, c='orange', marker='.')
    plt.title("Source background")
    plt.subplot(424)
    plt.ylim(top=1.05 * ylim)
    # plt.stem(x2)
    plt.scatter(np.arange(Ngrid), x2, c='orange', marker='.')
    plt.title("Recovered background")

    ylim = max(x.max(), (x1 + x2).max())
    plt.subplot(425)
    plt.ylim(top=1.05 * ylim)
    plt.stem(x)
    plt.scatter(np.arange(Ngrid), background, c='orange', marker='.', zorder=9)
    plt.title("Source signal")
    plt.subplot(426)
    plt.ylim(top=1.05 * ylim)
    plt.stem(x1 + x2)
    plt.scatter(np.arange(Ngrid), x2, c='orange', marker='.', zorder=9)
    plt.title("Recovered signal")

    # measurement fidelity

    measx1 = np.convolve(np.pad(x1, (kernel_width//2, kernel_width//2), mode='wrap'), kernel_measurement, mode='valid')
    measx2 = np.convolve(np.pad(x2, (kernel_width//2, kernel_width//2), mode='wrap'), kernel_measurement, mode='valid')
    sol_meas = (measx1 + measx2)[ds_factor//2::ds_factor]

    ylim = max(y.max(), sol_meas.max())
    plt.subplot(427)
    plt.ylim(top=1.05 * ylim)
    plt.stem(y)
    plt.title("Measurements")
    plt.subplot(428)
    plt.ylim(top=1.05 * ylim)
    plt.stem(sol_meas)
    plt.title("Measurements on the solution")

    plt.show()

    repr_std = 1.5
    representation_kernel = 1/(np.sqrt(2 * np.pi * repr_std**2)) * np.exp(-0.5 * np.arange(-3 * repr_std, 3 * repr_std + 1)**2 / repr_std**2)

    fig = plt.figure(figsize=(12, 11))
    plt.suptitle("Foreground representation")
    repr_source = np.convolve(img, representation_kernel, mode='same')
    repr_recovered = np.convolve(x1, representation_kernel, mode='same')
    ylim = max(repr_source.max(), repr_recovered.max())
    axes = fig.subplots(2, 1, sharex=True)
    ax = axes[0]
    ax.set_ylim(top=1.05 * ylim)
    # ax.stem(repr_source)
    ax.plot(np.arange(Ngrid), repr_source, c='orange', marker='.')
    ax.set_title("Source foreground")
    ax = axes[1]
    ax.set_ylim(top=1.05 * ylim)
    # ax.stem(repr_recovered)
    ax.plot(np.arange(Ngrid), repr_recovered, c='orange', marker='.')
    ax.set_title("Recovered foreground")
    plt.show()

    print(f"Relative L2 error on the foreground: {np.linalg.norm(repr_recovered - repr_source)/np.linalg.norm(repr_source):.2f}")

    l1_value = lambda1 * np.abs(x1).sum()
    print(f"Value of the foreground regularization at convergence: {l1_value:.3e}")
    l2_value = (np.convolve(x2, kernel_regul, mode='same')**2).sum()
    print(f"Approximate value of the background regularization at convergence: {lambda2 * l2_value:.3e}")
    data_fid_val = 0.5 * np.linalg.norm(y - sol_meas)**2
    print(f"Approximate value of the data fidelity at convergence: {data_fid_val:.3e}")

    #
    # vmin, vmax = min([y.min(), x1.min(), x2.min(), (x1+x2).min()]), max([y.max(), x1.max(), x2.max(), (x1+x2).max()])
    # fig = plt.figure(figsize=(10, 10))
    # axes = fig.subplots(2, 2, sharex=True, sharey=True)
    # for ax,  im, title in zip(axes.ravel(), [img, noise, x1, x2], ["Source image", "Noise", "Sparse component", "Smooth component"]):
    #     ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
    #     ax.set_title(title)
    # plt.suptitle(fr"Decoupled approach: $\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {pgd_time:.2f}s")
    # plt.show()
    #
    # plt.figure()
    # maxi = max(img.max(), x1.max())
    # plt.scatter(img[img != 0], x1[img != 0], label="True positive", color='green', marker='+')
    # plt.scatter(img[np.logical_and(x1 != 0, img==0)], x1[np.logical_and(x1 != 0, img==0)], label="False positive", color='orange', marker='+')
    # plt.axis((-0.5, maxi, -1, maxi))
    # plt.plot([-1, maxi], [-1, maxi], color='k', ls='--')
    # plt.hlines(0, -1, maxi, color='grey', ls='--')
    # plt.vlines(0, -1, maxi, color='grey', ls='--')
    # plt.legend()
    # plt.title("QQ-plot of the recovered intensities")
    # plt.suptitle("Decoupled approach")
    # plt.show()
    #
    # #--------------------------------------------------
    # # coupled approach
    # if do_comparison:
    #     coupled_df = 0.5 * pxop.SquaredL2Norm(img.size).asloss(y.ravel()) * pxop.stack([fOp, fOp], axis=1)
    #     smooth_regul = .5 * lambda2 * pxop.SquaredL2Norm(y.size) * pxop.stack([pxop.NullOp((y.size, y.size)), pxop.IdentityOp(y.size)], axis=1)
    #     sparse_regul = pxop.stack([lambda1 * pxop.PositiveL1Norm(y.size), pxop.NullOp((1, y.size))], axis=1)
    #
    #     print("Coupled solving...")
    #     solver_coupled = pxls.PGD(f=coupled_df + smooth_regul, g=sparse_regul, show_progress=False)
    #     start = time.time()
    #     solver_coupled.fit(x0=np.hstack([np.zeros(y.size), y.ravel()]), stop_crit=stop_crit)  # y.ravel()
    #     coupled_solving_time = time.time() - start
    #     xc = solver_coupled.solution()
    #     x1c, x2c = xc[:y.size].reshape(y.shape), xc[y.size:].reshape(y.shape)
    #
    #     vmin, vmax = min([y.min(), x1c.min(), x2c.min(), img.min()]), max([y.max(), x1c.max(), x2c.max(), img.max()])
    #     fig = plt.figure(figsize=(10, 10))
    #     axes = fig.subplots(2, 2, sharex=True, sharey=True)
    #     for ax, im, title in zip(axes.ravel(), [img, y, x1c, x2c],
    #                              ["Source image", "Noisy image", "Sparse component", "Smooth component"]):
    #         ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
    #         ax.set_title(title)
    #     plt.suptitle(
    #         fr"Coupled approach: $\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {coupled_solving_time:.2f}s")
    #     plt.show()
    #
    # print("Decoupled approach:")
    # print(f"\tRecovered sources: {np.count_nonzero(x1)}/{np.count_nonzero(img)}")
    # print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1[img != 0])}/{np.count_nonzero(x1)}")
    # print(f"\tRMSE on the noise: {np.sqrt(np.mean((noise - x2)**2)):.2f}")
    # print(f"\tReconstruction time: {pgd_time:.2f}s")
    # # print(f"\tPSNR sum: {20 * np.log10(np.max(y) / np.std(x1 + x2 - y)):.2f} dB")
    # # print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1 - img)):.2f} dB")
    # print(f"\tFinal value of the objective: {coupled_df(np.hstack(np.hstack([x1.ravel(), x2.ravel()])))[0] + smooth_regul(np.hstack([x1.ravel(), x2.ravel()]))[0] + sparse_regul(np.hstack([x1.ravel(), x2.ravel()]))[0]:.2f}")
    #
    # if do_comparison:
    #     print("Coupled approach:")
    #     print(f"\tRecovered sources: {np.count_nonzero(x1c)}/{np.count_nonzero(img)}")
    #     print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1c[img != 0])}/{np.count_nonzero(x1c)}")
    #     print(f"\tRMSE on the noise: {np.sqrt(np.mean((noise - x2c)**2)):.2f}")
    #     print(f"\tReconstruction time: {coupled_solving_time:.2f}s")
    #     # print(f"\tPSNR sum: {20 * np.log10(np.max(y) / np.std(x1c + x2c - y)):.2f} dB")
    #     # print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1c - img)):.2f} dB")
    #     print(f"\tFinal value of the objective: {coupled_df(xc)[0] + smooth_regul(xc)[0] + sparse_regul(xc)[0]:.2f}")
    #
