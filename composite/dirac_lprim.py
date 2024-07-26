"""
Similar experiment than the checkboard, but with a sparse image as the foreground
Noise is simple integration of a white noise
No measurement operator, strong Dirac impulses
Smooth penalty component: square root of laplacian
Comparative reconstruction: composite approach with decoupled and coupled solvers, and no composite approach
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pyxu.operator as pxop
import pyxu.opt.solver as pxls

from pyxu.opt.stop import RelError, MaxIter
from pyxu.abc import QuadraticFunc
from composite.smooth_noise import laplace_primitive_noise, square_laplace_primitive_noise

seed = None
psnrdb = 0
N = 128
sparsity = 2e-3

lambda1_factor = .09
lambda2 = .001
lambda_gl = 1e-2  # generalized LASSO
eps = 1e-5

do_comparison = True

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    rng = np.random.default_rng(seed=seed)

    img = np.zeros((N, N))
    Neff = int(.8 * N)
    k = int(N**2 * sparsity)
    idx = rng.choice(Neff ** 2, k, replace=False)
    indices = [arr + int(.1 * N) for arr in np.unravel_index(idx, (Neff, Neff))]
    img.ravel()[np.ravel_multi_index(indices, (N, N))] = rng.uniform(1, 10, k)

    # PSNR : 10 * np.log10(max(img)**2 / np.std(noise)**2) = 20 * log10(max(img) / std(noise))
    noise = laplace_primitive_noise(N, seed=seed, mu=0, sigma=5)
    sigma = np.max(img) * 10**(-psnrdb / 20)
    noise *= sigma / np.std(noise)
    exact_psnrdb = 20 * np.log10(np.max(img) / np.std(noise))
    print(f"Exact PSNR: {exact_psnrdb:.2f} dB")
    y = img + noise

    vmin, vmax = y.min(), y.max()
    plt.figure()
    plt.subplot(131)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(132)
    plt.imshow(noise, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(133)
    plt.imshow(img + noise, cmap='gray', vmin=vmin, vmax=vmax)
    plt.suptitle("Data: source image, noise, noisy image")
    plt.show()

    #--------------------------------------------------
    # Decoupled approach

    freq_response = -2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
                              np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2).ravel()  #.astype('complex128')
    diag = pxop.DiagonalOp(np.repeat(freq_response / (1 + lambda2 * freq_response), 2))
    dft = (1/N) * pxop.FFT(arg_shape=(N, N), real=True)  # normalisation to obtain unitary op
    Mlambda = lambda2 * dft.T * diag * dft
    Mlambda.lipschitz = 1 - 1/(1 + lambda2 * 8)  # explicit
    lambda1_max = np.abs(Mlambda.adjoint(y.ravel())).max()
    lambda1 = lambda1_factor * lambda1_max
    loss = QuadraticFunc((1, N**2), Q=Mlambda).asloss(y.ravel())
    loss.diff_lipschitz = Mlambda.lipschitz
    g = lambda1 * pxop.PositiveL1Norm(N**2)

    stop_crit = RelError(eps=eps, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

    pgd = pxls.PGD(f=loss, g=g, show_progress=False)
    start = time.time()
    pgd.fit(x0=np.zeros(N**2), stop_crit=stop_crit)
    solving_time = time.time() - start

    _, hist = pgd.stats()
    x1 = pgd.solution().reshape(y.shape)
    x2 = dft.adjoint(np.repeat(1 / (1 + lambda2 * freq_response), 2) *
                     dft((y - x1).ravel())).reshape(y.shape)

    vmin, vmax = min([y.min(), x1.min(), x2.min(), (x1+x2).min()]), max([y.max(), x1.max(), x2.max(), (x1+x2).max()])
    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax,  im, title in zip(axes.ravel(), [img, y, x1, x2], ["Source image", "Noisy image", "Sparse component", "Smooth component"]):
        ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(title)
    plt.suptitle(fr"Decoupled approach: $\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {solving_time:.2f}s")
    plt.show()

    plt.figure()
    maxi = max(img.max(), x1.max())
    plt.scatter(img[img != 0], x1[img != 0], label="True positive", color='green', marker='+')
    plt.scatter(img[np.logical_and(x1 != 0, img==0)], x1[np.logical_and(x1 != 0, img==0)], label="False positive", color='orange', marker='+')
    plt.axis((-0.5, maxi, -1, maxi))
    plt.plot([-1, maxi], [-1, maxi], color='k', ls='--')
    plt.hlines(0, -1, maxi, color='grey', ls='--')
    plt.vlines(0, -1, maxi, color='grey', ls='--')
    plt.legend()
    plt.title("QQ-plot of the recovered intensities")
    plt.show()

    if do_comparison:
        #--------------------------------------------------
        # Coupled approach
        sqrt_lap = dft.T * pxop.DiagonalOp(np.repeat(np.sqrt(freq_response), 2)) * dft
        data_fid = .5 * pxop.SquaredL2Norm(y.size).asloss(y.ravel()) * pxop.stack([pxop.IdentityOp(y.size), pxop.IdentityOp(y.size)], axis=1)
        smooth_regul = .5 * lambda2 * pxop.SquaredL2Norm(y.size) * pxop.stack([pxop.NullOp((y.size, y.size)), sqrt_lap], axis=1)
        sparse_regul = pxop.stack([lambda1 * pxop.PositiveL1Norm(y.size), pxop.NullOp((1, y.size))], axis=1)
        pos_coupled = pxop.stack([pxop.PositiveOrthant(dim=y.size), pxop.NullOp((1, y.size))], axis=1)

        solver_coupled = pxls.PGD(f=data_fid + smooth_regul, g=sparse_regul, show_progress=True, verbosity=100)
        start = time.time()
        solver_coupled.fit(x0=np.hstack([np.zeros(y.size), y.ravel()]), stop_crit=stop_crit)  # y.ravel()
        coupled_solving_time = time.time() - start
        xc = solver_coupled.solution()
        x1c, x2c = xc[:y.size].reshape(y.shape), xc[y.size:].reshape(y.shape)

        vmin, vmax = min([y.min(), x1c.min(), x2c.min(), img.min()]), max([y.max(), x1c.max(), x2c.max(), img.max()])
        fig = plt.figure(figsize=(10, 10))
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        for ax, im, title in zip(axes.ravel(), [img, y, x1c, x2c], ["Source image", "Noisy image", "Sparse component", "Smooth component"]):
            ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(title)
        plt.suptitle(fr"Coupled approach: $\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {coupled_solving_time:.2f}s")
        plt.show()

        #--------------------------------------------------
        # Simple LASSO solver
        lambda_factor = .1
        lambda_max = np.abs(y).max()
        lambda_gl = lambda_factor * lambda_max

        gl_data_fid = .5 * pxop.SquaredL2Norm(y.size).asloss(y.ravel())
        gl_regul = lambda_gl * pxop.PositiveL1Norm(y.size)

        stop_crit = RelError(eps=1e-2 * eps, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

        solver_gl = pxls.PGD(f=gl_data_fid, g=gl_regul, show_progress=False)
        start = time.time()
        solver_gl.fit(x0=np.zeros(y.size), stop_crit=stop_crit)  # y.ravel()
        gl_solving_time = time.time() - start
        x1gl = solver_gl.solution().reshape(y.shape)

        vmin, vmax = min([y.min(), x1gl.min(), (y-x1gl).min(), img.min()]), max([y.max(), x1gl.max(), (y-x1gl).max(), img.max()])
        fig = plt.figure(figsize=(10, 10))
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        for ax, im, title in zip(axes.ravel(), [img, y, x1gl, y-x1gl], ["Source image", "Noisy image", "Sparse component", "Residual"]):
            ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
        plt.suptitle(fr"Non-Composite GenLASSO: $\lambda$: {lambda_gl:.3f}, Solved in {gl_solving_time:.2f}s")
        plt.show()

    # todo:
    #  - metrics
    # -> Looks super promising as well!


    # Metrics: number of sources recovered, qqplot on intensity, RMSE on the noise
    print("Decoupled approach:")
    print(f"\tRecovered sources: {np.count_nonzero(x1)}/{np.count_nonzero(img)}")
    print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1[img != 0])}/{np.count_nonzero(x1)}")
    print(f"\tRMSE on the noise: {np.sqrt(np.mean((noise - x2)**2)):.2f}")
    print(f"\tReconstruction time: {solving_time:.2f}s")
    print(f"\tPSNR sum: {20 * np.log10(np.max(y) / np.std(x1 + x2 - y)):.2f} dB")
    print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1 - img)):.2f} dB")

    if do_comparison:
        print(
            f"\tFinal value of the objective: {data_fid(np.hstack(np.hstack([x1.ravel(), x2.ravel()])))[0] +smooth_regul(np.hstack([x1.ravel(), x2.ravel()]))[0] + sparse_regul(np.hstack([x1.ravel(), x2.ravel()]))[0]:.2f}")

        print("Coupled approach:")
        print(f"\tRecovered sources: {np.count_nonzero(x1c)}/{np.count_nonzero(img)}")
        print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1c[img != 0])}/{np.count_nonzero(x1c)}")
        print(f"\tRMSE on the noise: {np.sqrt(np.mean((noise - x2c)**2)):.2f}")
        print(f"\tReconstruction time: {coupled_solving_time:.2f}s")
        print(f"\tPSNR sum: {20 * np.log10(np.max(y) / np.std(x1c + x2c - y)):.2f} dB")
        print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1c - img)):.2f} dB")
        print(f"\tFinal value of the objective: {data_fid(xc)[0] + smooth_regul(xc)[0] + sparse_regul(xc)[0]:.2f}")

        print("Non-composite GenLASSO:")
        print(f"\tRecovered sources: {np.count_nonzero(x1gl)}/{np.count_nonzero(img)}")
        print(f"\t\tAmong them, correctly placed: {np.count_nonzero(x1gl[img != 0])}/{np.count_nonzero(x1gl)}")
        print(f"\tRMSE on the noise: {np.sqrt(np.mean((noise - x1gl)**2)):.2f}")
        print(f"\tReconstruction time: {gl_solving_time:.2f}s")
        print(f"\tPSNR sparse: {20 * np.log10(np.max(img) / np.std(x1gl - img)):.2f} dB")
