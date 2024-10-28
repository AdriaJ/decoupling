"""
Continuous-domain simulation.
Forward operator is convolution with a Gaussian kernel, sampled on a coarse grid.
Image model is sparse Dirac impulses and background made out of Gaussian kernels.
No penalty operator, simply identity.

Fixed background intensity, varying foreground with fgbgR.
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
from matplotlib import use
use("Qt5Agg")

#image model
seed = 1
Ngrid = 800
ds_factor = 8
Nmeas = Ngrid // ds_factor
k = 10
fgbgR = 10.
ongrid = True

# measurement model
kernel_std = 5  # Gaussian kernel std
kernel_width = 3 * 2 * kernel_std + 1  # Length of the Gaussian kernel
snrdb_meas = 20
norm_meas = (np.sqrt(2 * np.pi) * kernel_std)

# regularization
# kernel_std_regul = kernel_std / 2
# norm_regul = (np.sqrt(2 * np.pi) * kernel_std_regul)
lambda1_factor = 0.2
lambda2 = 1e-3
eps = 1e-5

blasso_factor = 0.3

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    rng = np.random.default_rng(seed=seed)

    img = np.zeros((Ngrid,))
    if ongrid:
        Neff = int(.6 * Ngrid)
        idx = rng.choice(Neff, k, replace=False)
        indices = idx + int(.2 * Ngrid)
        img[indices] = rng.uniform(1, fgbgR, k)

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

    # sbgrdb = 40
    # power_img = np.max(img)**2  # np.sum(img**2) / img.size
    # target_power_noise = power_img * 10**(-sbgrdb / 10)
    # sigma = np.sqrt(target_power_noise)
    # downscaling_factor = sigma / np.std(background)
    # downscaling_factor = 1
    # background *= downscaling_factor

    # img *= 0.5

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

    conv_bg = np.convolve(np.pad(bg_impulses, (width_meas_bg//2, width_meas_bg//2), mode='wrap'),
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


    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    # plt.stem(x)
    plt.stem(np.arange(img.shape[0])[img != 0], img[img != 0], basefmt="C0--")
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white', basefmt='C0--')
    plt.plot(np.arange(Ngrid), background)# c='orange',)
    plt.title("Original image (with background)")
    plt.subplot(212)
    plt.stem(y)
    plt.title("Noisy measurements")
    plt.show()

    #--------------------------------------------------

    diff_std2 = 2 * kernel_std**2
    norm_regul = np.sqrt(2 * np.pi * diff_std2)
    # xis = np.arange(ds_factor//2, Ngrid, ds_factor)
    diffs = np.arange(0, 4 * np.sqrt(diff_std2), ds_factor)
    diffs = np.hstack([-diffs[1:][::-1], diffs])
    kernel_regul = np.exp(-0.5 * diffs**2/diff_std2)
    # kernel_regul *= kernel_std**2 / ( kernel_std_regul**2 * np.sqrt(2 * np.pi * diff_std2) )
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
    # varphi_std2 = kernel_std**2 - 2 * kernel_std_regul**2
    # xis = np.arange(0, 3 * np.sqrt(varphi_std2) + 1)
    # varphi_sup = np.hstack([-xis[1:][::-1], xis])
    # varphi_kernel = np.exp(-0.5 * varphi_sup**2 / varphi_std2)
    # varphi_kernel *= (kernel_std/(2 * np.pi * (kernel_std_regul**2) * np.sqrt(varphi_std2)))
    tmp = np.zeros(Ngrid)
    tmp[ds_factor//2::ds_factor] = Mresiduals
    x2 = np.convolve(tmp, kernel_measurement, mode='same') / lambda2

    # ---------------

    print("BLASSO reconstruction...")
    lambda_max = np.abs(Hop.adjoint(y).ravel()).max()
    lambda_ = blasso_factor * lambda_max
    regul = lambda_ * pxop.PositiveL1Norm(Ngrid)
    loss = pxop.SquaredL2Norm(Nmeas).asloss(y) * Hop

    pgd = pxls.PGD(loss, g=regul, show_progress=False)
    start = time.time()
    pgd.fit(x0=np.zeros(Ngrid), stop_crit=stop_crit)
    blasso_time = time.time() - start

    x_blasso = pgd.solution()

    # ---------------

    print("Non-decoupled composite reconstruction...")

    Top = pxop.Convolve(
        arg_shape=Nmeas,
        kernel=[kernel_regul,],
        center=[kernel_regul.shape[0]//2,],
        mode="constant",
        enable_warnings=True,
    )

    ndcp_loss = .5 * pxop.SquaredL2Norm(Nmeas).asloss(y.ravel()) * pxop.hstack([Hop, Top]) + \
        lambda2 * pxop.hstack([pxop.NullFunc(Ngrid), QuadraticFunc((1, Nmeas), Q=Top)])
    ndcp_regul = lambda1 * pxop.hstack([pxop.PositiveL1Norm(Ngrid), pxop.NullFunc(Nmeas)])

    ndcp_stop = RelError(eps=eps, var="x", f= lambda v: v[:Ngrid], norm=2, satisfy_all=True,) & MaxIter(10)  # , f= lambda v: v[:Ngrid]


    ndcp_pgd = pxls.PGD(ndcp_loss, g=ndcp_regul, show_progress=False)
    start = time.time()
    ndcp_pgd.fit(x0=np.zeros(Ngrid + Nmeas), stop_crit=stop_crit)
    ndcp_time = time.time() - start

    ndcp_sol, ndcp_hst = ndcp_pgd.stats()

    x1_ndcp = ndcp_sol['x'][:Ngrid]
    x2_innovations_ndcp = ndcp_sol['x'][Ngrid:]
    x2_ndcp = np.zeros(Ngrid)
    x2_ndcp[ds_factor//2::ds_factor] = x2_innovations_ndcp
    x2_ndcp = np.convolve(np.pad(x2_ndcp, (kernel_width//2, kernel_width//2), mode='wrap'),
                          kernel_measurement, mode='valid')
    # print(np.allclose(x1_ndcp, 0))  # make sure the solution is non null

    # ---------------

    plt.figure(figsize=(12, 11))
    plt.suptitle(rf"$\lambda_1$ factor : {lambda1:.2e}, $\lambda_2$ : {lambda2:.2e}")
    ylim = max(img.max(), x1.max())
    # plt.subplot(421)
    plt.subplot(321)
    plt.ylim(top=1.05 * ylim)
    plt.stem(np.arange(img.shape[0])[img != 0], img[img != 0])
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white')
    plt.title("Source foreground")
    # plt.subplot(422)
    plt.subplot(322)
    plt.ylim(top=1.05 * ylim)
    plt.stem(np.arange(x1.shape[0])[x1 != 0], x1[x1 != 0])
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white')
    plt.title("Recovered foreground")

    ylim = max(background.max(), x2.max())
    # plt.subplot(423)
    plt.subplot(323)
    plt.ylim(top=1.05 * ylim)
    # plt.stem(background)
    plt.plot(np.arange(Ngrid), background, c='orange',)  # marker='.')
    plt.title("Source background")
    # plt.subplot(424)
    plt.subplot(324)
    plt.ylim(top=1.05 * ylim)
    # plt.stem(x2)
    plt.plot(np.arange(Ngrid), x2, c='orange',)  # marker='.')
    plt.title("Recovered background")

    # ylim = max(x.max(), (x1 + x2).max())
    # plt.subplot(425)
    # plt.ylim(top=1.05 * ylim)
    # plt.stem(img)
    # plt.plot(np.arange(Ngrid), background, c='orange', zorder=9)  # , marker='.'
    # plt.title("Source signal")
    # plt.subplot(426)
    # plt.ylim(top=1.05 * ylim)
    # plt.stem(x1)
    # plt.plot(np.arange(Ngrid), x2, c='orange', zorder=9)  # , marker='.'
    # plt.title("Recovered signal")

    # measurement fidelity

    measx1 = np.convolve(np.pad(x1, (kernel_width//2, kernel_width//2), mode='wrap'), kernel_measurement, mode='valid')
    measx2 = np.convolve(np.pad(x2, (kernel_width//2, kernel_width//2), mode='wrap'), kernel_measurement, mode='valid')
    sol_meas = (measx1 + measx2)[ds_factor//2::ds_factor]

    ylim = max(y.max(), sol_meas.max())
    # plt.subplot(427)
    plt.subplot(325)
    plt.ylim(top=1.05 * ylim)
    plt.stem(y)
    plt.title("Measurements")
    # plt.subplot(428)
    plt.subplot(326)
    plt.ylim(top=1.05 * ylim)
    plt.stem(sol_meas)
    plt.title("Measurements on the solution")

    plt.show()

    plt.figure(figsize=(6, 11))
    plt.subplot(311)
    ylim = max(img.max(), x1.max())
    plt.ylim(top=1.05 * ylim)
    plt.stem(np.arange(x1_ndcp.shape[0])[x1_ndcp != 0], x1_ndcp[x1_ndcp != 0])
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white')
    plt.title("Recovered foreground (non-decoupled)")
    plt.subplot(312)
    ylim = max(background.max(), x2.max())
    plt.ylim(top=1.05 * ylim)
    plt.plot(np.arange(Ngrid), x2_ndcp, c='orange',)  # marker='.')
    plt.title("Recovered background (non-decoupled)")
    plt.show()


    plt.figure(figsize=(6, 6))
    plt.stem(x_blasso)
    plt.title("Reconstruction BLASSO")
    plt.show()
    # plt.scatter(np.arange(Ngrid), x_blasso, label="BLASSO")

    repr_std = 1.5
    representation_kernel = 1/(np.sqrt(2 * np.pi * repr_std**2)) * np.exp(-0.5 * np.arange(-3 * repr_std, 3 * repr_std + 1)**2 / repr_std**2)

    fig = plt.figure(figsize=(16, 16))
    plt.suptitle("Foreground representation: convolution with a narrow Gaussian kernel")
    repr_source = np.convolve(img, representation_kernel, mode='same')
    repr_recovered = np.convolve(x1, representation_kernel, mode='same')
    repr_ndcp = np.convolve(x1_ndcp, representation_kernel, mode='same')
    repr_blasso = np.convolve(x_blasso, representation_kernel, mode='same')
    ylim = max(repr_source.max(), repr_recovered.max(), repr_ndcp.max())
    axes = fig.subplots(2, 2, sharex=True)
    ax = axes.ravel()[0]
    ax.set_ylim(top=1.05 * ylim)
    ax.plot(np.arange(Ngrid), repr_source, c='orange', marker='.')
    ax.set_title("Source foreground")
    ax = axes.ravel()[1]
    ax.set_ylim(top=1.05 * ylim)
    ax.plot(np.arange(Ngrid), repr_recovered, c='orange', marker='.')
    ax.set_title("Recovered foreground")
    ax = axes.ravel()[2]
    ax.set_ylim(top=1.05 * ylim)
    ax.plot(np.arange(Ngrid), repr_ndcp, c='orange', marker='.')
    ax.set_title("Non-decoupled foreground")
    ax = axes.ravel()[3]
    ax.set_ylim(top=1.05 * ylim)
    ax.plot(np.arange(Ngrid), repr_blasso, c='orange', marker='.')
    ax.set_title("BLASSO foreground")
    plt.show()

    # ------------------------------------------------

    print(f"Reconstruction times:")
    print(f"\tDecoupled: {pgd_time:.2f}s")
    print(f"\tNon-decoupled: {ndcp_time:.2f}s")
    print(f"\tBLASSO: {blasso_time:.2f}s")

    print(f"Relative L2 error on the foreground:")
    print(f"\tComposite: {np.linalg.norm(repr_recovered - repr_source)/np.linalg.norm(repr_source):.2f}")
    print(f"\tNon-decoupled: {np.linalg.norm(repr_ndcp - repr_source)/np.linalg.norm(repr_source):.2f}")
    print(f"\tBLASSO: {np.linalg.norm(repr_blasso - repr_source)/np.linalg.norm(repr_source):.2f}")


    l1_value = lambda1 * np.abs(x1).sum()
    print(f"Value of the foreground regularization at convergence: {l1_value:.3e}")
    l2_value = (np.convolve(x2, kernel_regul, mode='same')**2).sum()
    print(f"Approximate value of the background regularization at convergence: {lambda2 * l2_value:.3e}")
    data_fid_val = 0.5 * np.linalg.norm(y - sol_meas)**2
    print(f"Approximate value of the data fidelity at convergence: {data_fid_val:.3e}")

    # Wasserstein distance between simulated source and sparse reconstruction, using scipy
    from scipy.stats import wasserstein_distance
    img_sum, x1_sum = img.sum(), x1.sum()
    print(f"Sum of source: {img_sum:.3f}, sum of recovered: {x1_sum:.3f},"
          f"sum of non-decoupled: {x1_ndcp.sum():.3f} , sum of BLASSO: {x_blasso.sum():.3f}")
    print(f"Sum of source after convolution: {repr_source.sum():.3f}, sum of recovered: {repr_recovered.sum():.3f},"
          f"sum of non-decoupled: {repr_ndcp.sum():.3f}, sum of BLASSO: {repr_blasso.sum():.3f}")

    wass_dist = wasserstein_distance(np.arange(Ngrid), np.arange(Ngrid), img/img_sum, x1/x1_sum)
    print(f"Wasserstein distance between source and recovered: {wass_dist:.3f}")
    wass_dist_ndcp = wasserstein_distance(np.arange(Ngrid), np.arange(Ngrid), img/img_sum, x1_ndcp/x1_ndcp.sum())
    print(f"Wasserstein distance between source and non-decoupled: {wass_dist_ndcp:.3f}")
    wass_dist_blasso = wasserstein_distance(np.arange(Ngrid), np.arange(Ngrid), img/img_sum, x_blasso/x_blasso.sum())
    print(f"Wasserstein distance between source and BLASSO: {wass_dist_blasso:.3f}")

    #Wasserstein distance after convolution
    wass_dist_repr = wasserstein_distance(np.arange(Ngrid), np.arange(Ngrid), repr_source, repr_recovered)
    print(f"Wasserstein distance between source and recovered (after convolution): {wass_dist_repr:.3f}")
    wass_dist_repr_ndcp = wasserstein_distance(np.arange(Ngrid), np.arange(Ngrid), repr_source, repr_ndcp)
    print(f"Wasserstein distance between source and non-decoupled (after convolution): {wass_dist_repr_ndcp:.3f}")
    wass_dist_repr_blasso = wasserstein_distance(np.arange(Ngrid), np.arange(Ngrid), repr_source, repr_blasso)
    print(f"Wasserstein distance between source and BLASSO (after convolution): {wass_dist_repr_blasso:.3f}")
