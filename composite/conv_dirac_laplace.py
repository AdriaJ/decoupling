"""
Simulation of SMLM imaging:
Convolution forward operator (PSF approximated with a Gaussian kernel)
No sparse penalty operator
Laplacian smooth penalty operator
Data: sparse image with Dirac impulses
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
from composite.smooth_noise import square_laplace_primitive_noise


seed = 42
N = 128
sparsity = 2e-3
sbgrdb = 0

# measurement model
kernel_std = 4  # Gaussian kernel std
kernel_width = 3 * 2 * kernel_std + 1  # Length of the Gaussian kernel
snrdb_meas = 30

# reconstruction
lambda1_factor = .15  # 0.02 for regular background, .05 for positive background
lambda2 = .1
eps = 5e-5
l2op = "laplacian"  # "gaussian", "laplacian"
sigmal2 = 8
widthl2 = 3 * 2 * sigmal2 + 1

do_comparison = True
positive_background = True
gaussian_background = True

cmap = 'bwr'

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    rng = np.random.default_rng(seed=seed)

    # image synthesis
    img = np.zeros((N, N))
    Neff = int(.8 * N)
    k = int(N**2 * sparsity)
    idx = rng.choice(Neff ** 2, k, replace=False)
    indices = [arr + int(.1 * N) for arr in np.unravel_index(idx, (Neff, Neff))]
    img.ravel()[np.ravel_multi_index(indices, (N, N))] = rng.uniform(1, 10, k)

    # background synthesis
    if gaussian_background:
        background = np.zeros((N, N))
        idx = rng.choice(Neff ** 2, k, replace=False)
        indices = [arr + int(.1 * N) for arr in np.unravel_index(idx, (Neff, Neff))]
        background.ravel()[np.ravel_multi_index(indices, (N, N))] = rng.uniform(1, 5, k)
        kernel_std_bg = 2 * kernel_std
        kernel_width_bg = 3 * 2 * kernel_std_bg + 1
        kernel_bg_1d = (1 / (2 * np.pi * kernel_std_bg ** 2)) * np.exp(
            -0.5 * ((np.arange(kernel_width_bg) - (kernel_width_bg - 1) / 2) ** 2) / (kernel_std_bg ** 2))
        kernel_bg_1d = kernel_bg_1d / kernel_bg_1d.sum()
        kernel_bg_2d = kernel_bg_1d[:, None] * kernel_bg_1d[None, :]
        background = sig.fftconvolve(background, kernel_bg_2d, mode='same')
    else:
        background = square_laplace_primitive_noise(N, seed=seed, mu=0, sigma=1)
    if positive_background:
        background = np.abs(background)
    power_img = np.sum(img**2) / img.size
    target_power_noise = power_img * 10**(-sbgrdb / 10)
    sigma = np.sqrt(target_power_noise)
    background *= sigma / np.std(background)

    x = img + background

    # vmax = max(x.min(), x.max(), key=abs)
    # vmin = - vmax
    # plt.figure(figsize=(17, 5))
    # plt.subplot(131)
    # plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.subplot(132)
    # plt.imshow(background, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.subplot(133)
    # plt.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.suptitle("Data: source image, noise, noisy image")
    # plt.show()

    # convolution forward operator
    kernel_1d = (1 / (2 * np.pi * kernel_std ** 2)) * np.exp(
        -0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / (kernel_std ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    fOp = pxop.Convolve(
        arg_shape=img.shape,
        kernel=[kernel_1d, kernel_1d],
        center=[kernel_width // 2, kernel_width // 2],
        mode="wrap",  # constant
        enable_warnings=True,
    )
    fOp.lipschitz = fOp.estimate_lipschitz(method='svd', tol=1e-3)

    noiseless_y = fOp(x.ravel()).reshape(img.shape)
    sigma_noise = np.linalg.norm(noiseless_y)/N * 10**(-snrdb_meas / 20)
    noise_meas = rng.normal(0, sigma_noise, img.shape)
    y = noiseless_y + noise_meas

    if positive_background:
        cmap = 'gray_r'

    plt.figure(figsize=(17, 5))
    plt.subplot(131)
    vlim = max(x.min(), x.max(), key=abs)
    vmin = 0 if positive_background else -vlim
    plt.imshow(x, cmap=cmap, interpolation='none', vmin=vmin, vmax=vlim)
    plt.title("Original image (with background noise)")
    plt.colorbar()
    plt.subplot(132)
    vlim = max(noiseless_y.min(), noiseless_y.max(), y.min(), y.max(), key=abs)
    vmin = 0 if positive_background else -vlim
    plt.imshow(noiseless_y, cmap=cmap, interpolation='none', vmin=vmin, vmax=vlim)
    plt.title("Noiseless measurements")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(y, cmap=cmap, interpolation='none', vmin=vmin, vmax=vlim)
    plt.title("Noisy measurements")
    plt.colorbar()
    plt.show()

    #--------------------------------------------------
    # computation of Mlambda2
    l2op_freq_response = None
    if l2op == "laplacian":
        l2op_freq_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
                                 np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2).ravel()
    elif l2op == "gaussian":
        kernel_l2op = (1 / (2 * np.pi * sigmal2 ** 2)) * np.exp(
                        -0.5 * ((np.arange(widthl2) - (widthl2 - 1) / 2) ** 2) / (sigmal2 ** 2))
        kernel_l2op = kernel_l2op / kernel_l2op.sum()
        kpad = np.zeros(N)
        kpad[:widthl2 // 2 + 1] = kernel_l2op[widthl2 // 2:widthl2]
        kpad[-widthl2 // 2 + 1:] = kernel_l2op[:widthl2 // 2]
        freq1d_l2 = sfft.fft(kpad).real  # imag is zero
        l2op_freq_response = (freq1d_l2[:, None] * freq1d_l2[None, :]).ravel()

    kernel_padded = np.zeros(N)
    kernel_padded[:kernel_width//2 + 1] = kernel_1d[kernel_width//2:kernel_width]
    kernel_padded[-kernel_width//2 + 1:] = kernel_1d[:kernel_width//2]
    freq_response_1d = sfft.fft(kernel_padded).real  # imag is zero
    freq_response_2d = (freq_response_1d[:, None] * freq_response_1d[None, :]).ravel()

    # A and L_2 commute: the operator Lambda_2 is simply L2^T L2
    # Mlambda2 in the frequency domain: L2 ** 2 / (A**2 + lambda2 L2**2)

    diag = pxop.DiagonalOp(np.repeat(l2op_freq_response**2 / (freq_response_2d**2 + lambda2 * l2op_freq_response**2), 2))
    dft = (1/N) * pxop.FFT(arg_shape=(N, N), real=True)  # normalisation to obtain unitary op
    Mlambda = lambda2 * dft.T * diag * dft

    lambda1max = np.abs(fOp.adjoint(Mlambda(y.ravel()))).max()
    lambda1 = lambda1_factor * lambda1max

    loss = QuadraticFunc((1, N ** 2), Q=Mlambda).asloss(y.ravel()) * fOp
    # loss.diff_lipschitz = loss.estimate_diff_lipschitz(method='svd')  # Mlambda.lipschitz  # fOp.lipschitz = 1.

    regul = lambda1 * pxop.PositiveL1Norm(N ** 2)

    stop_crit = RelError(eps=eps, var="x", f=None, norm=2, satisfy_all=True, ) & MaxIter(10)

    print("Decoupled solving...")
    pgd = pxls.PGD(loss, g=regul, show_progress=False)
    start = time.time()
    pgd.fit(x0=np.zeros(img.size), stop_crit=stop_crit)
    pgd_time = time.time() - start

    _, hist = pgd.stats()
    x1 = pgd.solution().reshape(y.shape)
    x2 = dft.adjoint(np.repeat(freq_response_2d / (freq_response_2d ** 2 + lambda2 * l2op_freq_response ** 2), 2) *
                     dft(y.ravel() - fOp(x1.ravel()))).reshape(y.shape)

    vmax = max([y.max(), x1.max(), x2.max(), (x1 + x2).max()])
    vmin = 0 if positive_background else -vmax
    if positive_background:
        cmap = 'gist_heat_r'  # 'gray_r'
    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, im, title in zip(axes.ravel(), [img, background, x1, x2],
                             ["Source image", "Noise", "Sparse component", "Smooth component"]):
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(title)
    plt.suptitle(
        fr"Decoupled approach: $\lambda_1$: {lambda1:.2e}, $\lambda_2$: {lambda2:.3f}, Solved in {pgd_time:.2f}s")
    plt.show()

    import scipy.signal as ss
    sigma = 1.
    repr_kernel = np.exp(-0.5 * np.arange(-4 * sigma, 4 * sigma + 1) ** 2 / sigma ** 2)
    repr_kernel /= repr_kernel.sum()
    kernel_2d = repr_kernel[:, None] * repr_kernel[None, :]

    cmap = 'gist_heat_r'
    fig = plt.figure(figsize=(7, 10))
    axes = fig.subplots(2, 1, sharex=True, sharey=True)
    for ax, im, title in zip(axes.ravel(), [img, x1],
                             ["Source image", "Noise", "Sparse component", "Smooth component"]):
        ax.imshow(ss.fftconvolve(im, kernel_2d, mode='same'), cmap=cmap, vmin=0., interpolation='none')
        ax.set_title(title)
    plt.suptitle(
        fr"Decoupled approach: $\lambda_1$: {lambda1:.2e}, $\lambda_2$: {lambda2:.3f}, Solved in {pgd_time:.2f}s")
    plt.show()

    #todo Simulate a different background model

    if do_comparison:
        l2Op = None
        if l2op == "laplacian":
            l2Op = pxop.Laplacian((N, N), mode='wrap')
        elif l2op == "gaussian":
            l2Op = pxop.Convolve(
                arg_shape=img.shape,
                kernel=[kernel_l2op, kernel_l2op],
                center=[widthl2 // 2, widthl2 // 2],
                mode="wrap",  # constant
                enable_warnings=True,
            )
            l2Op.lipschitz = l2Op.estimate_lipschitz(method='svd', tol=1e-3)
            print(np.allclose(l2Op.lipschitz, kernel_l2op.max()**2))

        coupled_df = 0.5 * pxop.SquaredL2Norm(img.size).asloss(y.ravel()) * fOp * pxop.stack([pxop.IdentityOp(N**2), pxop.IdentityOp(N**2)], axis=1)
        smooth_regul = .5 * lambda2 * pxop.SquaredL2Norm(y.size) * pxop.stack([pxop.NullOp((y.size, y.size)), l2Op], axis=1)
        sparse_regul = pxop.stack([lambda1 * pxop.PositiveL1Norm(y.size), pxop.NullOp((1, y.size))], axis=1)

        coupled_stop_crit = RelError(eps=eps, var="x",
                                     f=lambda u: u[:y.size],
                                     norm=2, satisfy_all=True, ) & MaxIter(10)

        print("Coupled solving...")
        solver_coupled = pxls.PGD(f=coupled_df + smooth_regul, g=sparse_regul, show_progress=False)
        start = time.time()
        solver_coupled.fit(x0=np.hstack([np.zeros(y.size), y.ravel()]), stop_crit=coupled_stop_crit)  # y.ravel()
        coupled_solving_time = time.time() - start
        xc = solver_coupled.solution()
        x1c, x2c = xc[:y.size].reshape(y.shape), xc[y.size:].reshape(y.shape)

        vmax = max([y.max(), x1c.max(), x2c.max(), img.max()])
        vmin = 0 if positive_background else -vmax
        if positive_background:
            cmap = 'gray_r'
        fig = plt.figure(figsize=(10, 10))
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        for ax, im, title in zip(axes.ravel(), [img, y, x1c, x2c],
                                 ["Source image", "Noisy image", "Sparse component", "Smooth component"]):
            ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(title)
        plt.suptitle(
            fr"Coupled approach: $\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {coupled_solving_time:.2f}s")
        plt.show()

    print("Decoupled approach:")
    print(f"\tRecovered sources: {np.count_nonzero(x1)}/{np.count_nonzero(img)}")
    print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1[img != 0])}/{np.count_nonzero(x1)}")
    print(f"\tRMSE on the noise: {np.sqrt(np.mean((background - x2)**2)):.2f}")
    print(f"\tReconstruction time: {pgd_time:.2f}s")
    # print(f"\tPSNR sum: {20 * np.log10(np.max(y) / np.std(x1 + x2 - y)):.2f} dB")
    # print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1 - img)):.2f} dB")
    print(f"\tFinal value of the objective: {coupled_df(np.hstack(np.hstack([x1.ravel(), x2.ravel()])))[0] + smooth_regul(np.hstack([x1.ravel(), x2.ravel()]))[0] + sparse_regul(np.hstack([x1.ravel(), x2.ravel()]))[0]:.2f}")

    if do_comparison:
        print("Coupled approach:")
        print(f"\tRecovered sources: {np.count_nonzero(x1c)}/{np.count_nonzero(img)}")
        print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1c[img != 0])}/{np.count_nonzero(x1c)}")
        print(f"\tRMSE on the noise: {np.sqrt(np.mean((background - x2c)**2)):.2f}")
        print(f"\tReconstruction time: {coupled_solving_time:.2f}s")
        # print(f"\tPSNR sum: {20 * np.log10(np.max(y) / np.std(x1c + x2c - y)):.2f} dB")
        # print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1c - img)):.2f} dB")
        print(f"\tFinal value of the objective: {coupled_df(xc)[0] + smooth_regul(xc)[0] + sparse_regul(xc)[0]:.2f}")

    # simple LASSO comparison

    lambda_factor = .15
    lambda_max = np.abs(fOp.adjoint(y.ravel())).max()
    lambda_lasso = lambda_factor * lambda_max

    data_fid_lasso = .5 * pxop.SquaredL2Norm(y.size).asloss(y.ravel()) * fOp
    regul_lasso = lambda_lasso * pxop.PositiveL1Norm(y.size)

    solver_lasso = pxls.PGD(f=data_fid_lasso, g=regul_lasso, show_progress=False)
    start = time.time()
    solver_lasso.fit(x0=np.zeros(y.size), stop_crit=stop_crit)
    lasso_solving_time = time.time() - start
    xlasso = solver_lasso.solution().reshape(y.shape)

    print(f"Reconstruction time for LASSO: {lasso_solving_time:.2f}s")

    vmin=0
    vmax = max([y.max(), xlasso.max(), img.max(), x1.max()])
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(img, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title("Source foreground")
    plt.subplot(222)
    plt.imshow(y, cmap=cmap, interpolation='none')
    plt.title("Noisy measurements")
    plt.subplot(223)
    plt.imshow(x1, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title("Composite decoupled reconstruction")
    plt.subplot(224)
    plt.imshow(xlasso, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title("LASSO reconstruction")
    plt.show()

    vmin=0
    cmap = 'gist_heat_r'
    conv_data = [ss.fftconvolve(img, kernel_2d, mode='same'), ss.fftconvolve(x1, kernel_2d, mode='same'), ss.fftconvolve(xlasso, kernel_2d, mode='same')]
    vmax = max([d.max() for d in conv_data])
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(conv_data[0], cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title("Source foreground")
    plt.subplot(222)
    plt.imshow(y, cmap=cmap, interpolation='none')
    plt.title("Noisy measurements")
    plt.subplot(223)
    plt.imshow(conv_data[1], cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title("Composite decoupled reconstruction")
    plt.subplot(224)
    plt.imshow(conv_data[2], cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title("LASSO reconstruction")
    plt.show()

    # Save images as pdf: img, background, y, x1, xlasso
    save = False
    if save:
        import os
        cmap = 'gist_heat_r'
        save_path = "/home/jarret/PycharmProjects/decoupling/composite/figures/gauss_conv"

        conv_data = [ss.fftconvolve(img, kernel_2d, mode='same'), ss.fftconvolve(x1, kernel_2d, mode='same'),
                     ss.fftconvolve(xlasso, kernel_2d, mode='same')]
        vmax = 1.25  # max(conv_data[0].max(), conv_data[1].max())

        for toplot, name in zip([conv_data[0], x, conv_data[1], conv_data[2]], ["fg", "fg+bg", "composite_reco_fg", "lasso_fg"]):
            plt.figure()
            plt.imshow(toplot, cmap=cmap, vmin=vmin, interpolation="none", vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            # plt.colorbar()
            # plt.show()
            plt.savefig(os.path.join(save_path, f"{name}.pdf"), dpi=600, bbox_inches='tight', pad_inches=.2)
            plt.close()

        for toplot, name in zip([background, y, ss.fftconvolve(y, kernel_2d, mode='same')], ["bg", "measurements", "conv_measurements"]):
            plt.figure()
            plt.imshow(toplot, cmap=cmap, vmin=vmin, interpolation="none", vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.show()
            # plt.savefig(os.path.join(save_path, f"{name}.pdf"), dpi=600, bbox_inches='tight', pad_inches=.2)
            # plt.close()




