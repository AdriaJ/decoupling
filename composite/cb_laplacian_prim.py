"""
Checkboard + (double) Laplacian primitve of a Gaussian noise
Different noise model and smooth component model
Comparative reconstruction between decoupled solving, coupled solving and non-composite approach (simple GenLASSO solver)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pyxu.operator as pxop
import pyxu.opt.solver as pxls
import pyxu.util as pxu

from pyxu.opt.stop import RelError, MaxIter
from pyxu.abc import QuadraticFunc
from composite.smooth_noise import laplace_primitive_noise, square_laplace_primitive_noise

import skimage as ski

seed = None
snrdb = 0

lambda1 = 5e-3
alpha = 40  # lambda2
lambda_gl = 1e-2  # generalized LASSO
eps = 1e-5

if __name__ == "__main__":
    lambda2 = alpha * lambda1

    if seed is None:
        seed = np.random.randint(1000)

    img = ski.data.checkerboard().astype(np.float64) / 255
    N = img.shape[0]//2  # //4
    img = img[:N, :N]
    lap = pxop.Laplacian((N, N), mode='wrap')

    # SNR = mean(img**2) / mean(noise**2), mean(noise**2) = sigma**2
    # SNRdb = 10 * log10(snr)
    power_img = np.sum(img**2) / img.size
    target_power_noise = power_img * 10**(-snrdb / 10)
    sigma = np.sqrt(target_power_noise)
    noise = square_laplace_primitive_noise(N, seed=seed, mu=0, sigma=1)
    noise *= sigma / np.std(noise)
    exact_snrdb = 10 * np.log10(power_img / np.std(noise)**2)
    print(f"Exact SNR: {exact_snrdb:.2f} dB")

    y = img + noise
    # y = 255. * (y - y.min())/(y.max() - y.min())

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

    freq_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
                              np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2).ravel()  #.astype('complex128')
    diag = pxop.DiagonalOp(np.repeat(freq_response**2 / (1 + lambda2 * freq_response**2), 2))
    dft = (1/N) * pxop.FFT(arg_shape=(N, N), real=True)  # normalisation to obtain unitary op
    Mlambda = lambda2 * dft.T * diag * dft
    Mlambda.lipschitz = 1 - 1/(1 + lambda2 * 8**2)  # explicit

    loss = QuadraticFunc((1, N**2), Q=Mlambda).asloss(y.ravel())
    loss.diff_lipschitz = Mlambda.lipschitz

    l21 = pxop.L21Norm(arg_shape=(2, *y.shape), l2_axis=(0, ))

    grad = pxop.Gradient(arg_shape=y.shape, diff_method="fd", scheme="forward", mode="symmetric", accuracy=1,)
    grad.lipschitz = grad.estimate_lipschitz(method='svd')

    stop_crit = RelError(eps=eps, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

    positivity = pxop.PositiveOrthant(dim=y.size)
    solver = pxls.PD3O(f=loss, g=positivity, h=lambda1 * l21, K=grad, verbosity=2_000)
    start = time.time()
    solver.fit(x0=y.ravel(), stop_crit=stop_crit)  # y.ravel()
    solving_time = time.time() - start

    _, hist = solver.stats()
    x1 = solver.solution().reshape(y.shape)
    x2 = dft.adjoint(np.repeat(1 / (1 + lambda2 * freq_response**2), 2) *
                     dft((y - x1).ravel())).reshape(y.shape)

    # plot the results
    g = grad(x1.ravel())
    vmin, vmax = min([y.min(), x1.min(), x2.min(), (x1+x2).min()]), max([y.max(), x1.max(), x2.max(), (x1+x2).max()])
    plt.figure(figsize=(14, 10))
    plt.subplot(231)
    plt.title("Gradient of the sparse component")
    plt.imshow(np.linalg.norm(np.stack([g[:x1.size], g[x1.size:]]), axis=0).reshape(y.shape))
    plt.colorbar(location='left')
    plt.subplot(232)
    plt.title("Reconstruction: Sparse component")
    plt.imshow(x1, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(233)
    plt.title("Reconstruction: Smooth component")
    plt.imshow(x2, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(234)
    plt.title("Source image")
    plt.imshow(img, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(235)
    plt.title("Noisy image (measurements)")
    plt.imshow(y, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(236)
    plt.title(f"Reconstruction: sum")
    plt.imshow(x1+x2, cmap='gray',vmin=vmin, vmax=vmax)
    plt.suptitle(fr"Decoupled approach: $\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {solving_time:.2f}s")
    plt.show()

    print(f"Value of the decoupled loss:")
    print(f"\tOn the measurements: {loss(y.ravel())[0]:.3e}")
    print(f"\tOn the source image: {loss(img.ravel())[0]:.3e}")
    print("Value fo the different terms of the objective function (decoupled solver):")
    print(f"\tData fidelity: {0.5 * np.linalg.norm(y - x1 - x2)**2:.3e}")
    print(f"\tSmooth penalty: {.5 * lambda2 * np.linalg.norm(lap(x2.ravel()))**2:.3e}")
    print(f"\tSparse penalty: {lambda1 * l21(grad(x1.ravel()))[0]:.3e}")
    print(f"\tTotal cost: {0.5 * np.linalg.norm(y - x1 - x2)**2 + lambda1 * l21(grad(x1.ravel()))[0] + .5 * lambda2 * np.linalg.norm(lap(x2.ravel()))**2:.3e}")

    print("Same value on the source image:")
    print(f"\tData fidelity: {0.5 * np.linalg.norm(y - img - noise)**2:.3e}")
    print(f"\tSmooth penalty: {.5 * lambda2 * np.linalg.norm(lap(noise.ravel()))**2:.3e}")
    print(f"\tSparse penalty: {lambda1 * l21(grad(img.ravel()))[0]:.3e}")


    mse = np.sum((x1 - img)**2)


    plt.figure()
    plt.scatter(hist['iteration'], hist['RelError[x]'], label='Stopping criterion', marker='.', s=10)
    plt.yscale('log')
    plt.title("Stopping criterion of the decoupled solver")
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.title("Histogram of the source image")
    plt.hist(img.ravel(), bins=100)
    plt.subplot(122)
    plt.title("Histogram of the noisy image")
    plt.hist(y.ravel(), bins=100)
    plt.show()


    # Coupled solver

    data_fid = .5 * pxop.SquaredL2Norm(y.size).asloss(y.ravel()) * pxop.stack([pxop.IdentityOp(y.size), pxop.IdentityOp(y.size)], axis=1)
    smooth_regul = .5 * lambda2 * pxop.SquaredL2Norm(y.size) * pxop.stack([pxop.NullOp((y.size, y.size)), lap], axis=1)
    sparse_regul = lambda1 * l21 * pxop.stack([grad, pxop.NullOp((2 * y.size, y.size))], axis=1)
    sparse_op = pxop.stack([grad, pxop.NullOp((2 * y.size, y.size))], axis=1)
    pos_coupled = pxop.stack([pxop.PositiveOrthant(dim=y.size), pxop.NullOp((1, y.size))], axis=1)

    # stop_crit = RelError(eps=1e-5, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

    coupled_stop_crit = RelError(eps=eps, var="x",
                                 f=lambda u: u[:y.size],
                                 norm=2, satisfy_all=True,) & MaxIter(10)

    solver_coupled = pxls.PD3O(f=data_fid + smooth_regul, g=pos_coupled, h=lambda1 * l21, K=sparse_op, verbosity=2_000)
    start = time.time()
    solver_coupled.fit(x0=np.hstack([y.ravel(), np.zeros(y.size)]), stop_crit=coupled_stop_crit)  # y.ravel()
    coupled_solving_time = time.time() - start
    xc = solver_coupled.solution()
    x1c, x2c = xc[:y.size].reshape(y.shape), xc[y.size:].reshape(y.shape)

    # Evaluate the objective function and the penalties on the solutions
    print("Value of the different terms of the objective function (coupled solver):")
    print(f"\tData fidelity: {data_fid(xc)[0]:.3e}")
    print(f"\tSmooth penalty: {smooth_regul(xc)[0]:.3e}")
    print(f"\tSparse penalty: {sparse_regul(xc)[0]:.3e}")
    print(f"\tTotal cost: {data_fid(xc)[0] + smooth_regul(xc)[0] + sparse_regul(xc)[0]:.3e}")

    # Show the solutions
    plt.figure(figsize=(14, 10))
    plt.subplot(231)
    plt.title("Gradient of the sparse component")
    plt.imshow(np.linalg.norm(np.stack([grad(x1c.ravel())[:x1c.size], grad(x1c.ravel())[x1c.size:]]), axis=0).reshape(y.shape))
    plt.colorbar(location='left')
    plt.subplot(232)
    plt.title("Reconstruction: Sparse component")
    plt.imshow(x1c, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(233)
    plt.title("Reconstruction: Smooth component")
    plt.imshow(x2c, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(234)
    plt.title("Source image")
    plt.imshow(img, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(235)
    plt.title("Noisy image (measurements)")
    plt.imshow(y, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(236)
    plt.title(f"Reconstruction: sum")
    plt.imshow(x1c+x2c, cmap='gray',vmin=vmin, vmax=vmax)
    plt.suptitle(fr"Coupled approach: Lambda1: {lambda1:.3f}, Lambda2: {lambda2:.3f}, Solved in {coupled_solving_time:.2f}s")
    plt.show()


    # Simple genLASSSO solver

    lambda_gl = 1e-1

    data_fid_gl = .5 * pxop.SquaredL2Norm(y.size).asloss(y.ravel())
    sparse_regul_gl = lambda1 * l21 * grad
    stop_crit = RelError(eps=1e-2 * eps, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

    solver_gl = pxls.PD3O(f=data_fid_gl, g=positivity, h=lambda_gl * l21, K=grad, verbosity=2_000, show_progress=False)
    start = time.time()
    solver_gl.fit(x0=y.ravel(), stop_crit=stop_crit)  # y.ravel()
    gl_solving_time = time.time() - start
    x1gl = solver_gl.solution().reshape(y.shape)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title("Gradient of the sparse component")
    plt.imshow(np.linalg.norm(np.stack([grad(x1gl.ravel())[:x1gl.size], grad(x1gl.ravel())[x1gl.size:]]), axis=0).reshape(y.shape))
    plt.colorbar(location='left')
    plt.subplot(222)
    plt.title("Reconstruction: Sparse component")
    plt.imshow(x1gl, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(223)
    plt.title("Source image")
    plt.imshow(img, cmap='gray',vmin=vmin, vmax=vmax)
    plt.subplot(224)
    plt.title("Noisy image (measurements)")
    plt.imshow(y, cmap='gray',vmin=vmin, vmax=vmax)
    plt.suptitle(fr"Generalized LASSO: Lambda: {lambda_gl:.3e}, Solved in {gl_solving_time:.2f}s")
    plt.show()
    print("Metrics for GenLASSO approach:")
    print(f"\tSolving time: {gl_solving_time:.2f}s")
    print(f"\tRMSE: {np.sqrt(np.sum((x1gl - img)**2))/N:.3e}")
    print(f"\tSNR: {10 * np.log10(np.sum(img**2) / np.sum((img - x1gl)**2)):.2f} dB")

    # produce some final comparison metrics
    print("Coupled approach:")
    print(f"\tSolving time: {coupled_solving_time:.2f}s")
    print(f"\tFinal value of the cost: {data_fid(xc)[0] + smooth_regul(xc)[0] + sparse_regul(xc)[0]:.3e}")
    print(f"\tRMSE source: {np.sqrt(np.sum((x1c - img)**2))/N:.3e}")
    print(f"\tRMSE noise: {np.sqrt(np.sum((x2c - noise)**2))/N:.3e}")
    print("Decoupled approach:")
    print(f"\tSolving time: {solving_time:.2f}s")
    print(f"\tFinal value of the cost: {0.5 * np.linalg.norm(y - x1 - x2)**2 + lambda1 * l21(grad(x1.ravel()))[0] + .5 * lambda2 * np.linalg.norm(lap(x2.ravel()))**2:.3e}")
    print(f"\tRMSE source: {np.sqrt(np.sum((x1 - img)**2))/N:.3e}")
    print(f"\tRMSE noise: {np.sqrt(np.sum((x2 - noise)**2))/N:.3e}")

    # compute the SNR between img and img - x1
    snr = 10 * np.log10(np.sum(img**2) / np.sum((img - x1)**2))
    print(f"SNR between the source image and the sparse reconstruction: {snr:.2f} dB")
    # snr_power = 20 * np.log10(np.std(img) / np.std(img - x1))
    # print(f"SNR power between the source image and the sparse reconstruction: {snr_power:.2f} dB")

    plt.figure()
    plt.title("Error map between source and sparse reconstruction")
    plt.imshow(np.abs(img-x1), cmap='gray', interpolation='none', vmin=0, vmax=1)
    plt.show()

    #todo introduce metrics: MSE, SNR

    # Save images as pdf: img, noise, img+noise+noise ??, x1, xigl
    save=False
    if save:
        import os

        for toplot, name in zip([img, noise, y, x1, x1gl], ["fg", "bg", "measurements", "composite_reco_fg", "gen_lasso_fg"]):
            plt.figure()
            plt.imshow(toplot, cmap='gray', vmin=vmin, vmax=vmax, interpolation="none")
            # plt.show()
            plt.savefig(os.path.join("/home/jarret/PycharmProjects/decoupling/composite/figures/checkboard", f"{name}.pdf"),
                        dpi=600, bbox_inches='tight', pad_inches=.2)
    # ========================================
    # =============== TESTS  =================
    # ========================================

    # # test correctness of the operator
    # lap = pxop.Laplacian((N, N), mode='wrap')
    # freq_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
    #                      np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2).ravel()
    # arr = np.random.randn(lap.shape[1])
    # tmp = pxu.view_as_complex(dft(arr))
    # res = dft.adjoint(pxu.view_as_real(freq_response * tmp))
    # print(np.allclose(lap(arr), res))  # True
    #
    # tmp2 = dft(arr)
    # res2 = dft.adjoint(np.repeat(freq_response, 2) * tmp2)
    # print(np.allclose(lap(arr), res2))  # True
    #
    #
    # op = dft.T * pxop.DiagonalOp(np.repeat(freq_response, 2)) * dft
    # np.allclose(lap.asarray(), op.asarray())
    # print(np.allclose(lap(arr), op(arr)))  # True

    # # test diagonal implementation of mlambda
    # N = 20
    # freq_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
    #                           np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2).ravel()  #.astype('complex128')
    # diag = pxop.DiagonalOp(np.repeat(freq_response**2 / (1 + lambda2 * freq_response**2), 2))
    # dft = (1/N) * pxop.FFT(arg_shape=(N, N), real=True)  # normalisation to obtain unitary op
    # Mlambda = lambda2 * dft.T * diag * dft
    #
    # arr = np.random.randn(dft.shape[1])
    #
    # lap = pxop.Laplacian((N, N), mode='wrap')
    # Lamb = lap * lap
    # mat = np.eye(N**2, N**2) + lambda2 * Lamb.asarray()
    # res = lambda2 * Lamb(np.linalg.inv(mat) @ arr)
    # print(np.allclose(Mlambda(arr), res))  # True

    # arr = np.random.randn(dft.shape[1])
    # print(np.allclose(dft.adjoint(dft(arr)), arr))
    # z = np.random.randn(dft.shape[0])  # ok if fft takes complex valued inputs
    # print(np.allclose(dft(dft.adjoint(z)), z))

    # # test if Mlambda is semi definite positive ? yes
    # arr = np.random.randn(Mlambda.shape[1])
    # print(np.dot(arr, Mlambda(arr)) >= 0)

    # # test asloss method
    # arr = np.random.randn(y.size)
    # print(np.allclose(loss(y.ravel() + arr)[0], arr @ Mlambda(arr)/2))  # True


    # ========================================
    # ========= Condat-Vu Decoupled Solver
    # ========================================

    # cv = pxls.CondatVu(f=loss, g=positivity, h=lambda1 * l21, K=grad, verbosity=500)
    # start = time.time()
    # cv.fit(x0=np.zeros(y.size), z0=np.zeros(grad.shape[0]), stop_crit=stop_crit & RelError(eps=eps, var="z", f=None, norm=2, satisfy_all=True,))  # y.ravel()
    # solving_time_cv = time.time() - start
    #
    # _, hist_cv = cv.stats()
    # x1_cv = cv.solution().reshape(y.shape)
    # x2_cv = dft.adjoint(np.repeat(freq_response / (1 + lambda2 * freq_response**2), 2) *
    #                     dft((y - x1_cv).ravel())).reshape(y.shape)

    # # produce the same pllots with x1_cv and x2_cv instead of x1 and x2
    # g_cv = grad(x1_cv.ravel())
    #
    # vmin, vmax = (min([y.min(), x1_cv.min(), x2_cv.min(), (x1_cv+x2_cv).min()]),
    #               max([y.max(), x1_cv.max(), x2_cv.max(), (x1_cv+x2_cv).max()]))
    #
    # plt.figure(figsize=(14, 10))
    # plt.subplot(231)
    # plt.title("Gradient of the sparse component")
    # plt.imshow(np.linalg.norm(np.stack([g[:x1.size], g[x1.size:]]), axis=0).reshape(y.shape))
    # plt.colorbar(location='left')
    # plt.subplot(232)
    # plt.title("Reconstruction: Sparse component")
    # plt.imshow(x1_cv, cmap='gray',vmin=vmin, vmax=vmax)
    # plt.subplot(233)
    # plt.title("Reconstruction: Smooth component")
    # plt.imshow(x2_cv, cmap='gray', vmin=vmin, vmax=vmax)
    # plt.subplot(234)
    # plt.title("Source image")
    # plt.imshow(img, cmap='gray',vmin=vmin, vmax=vmax)
    # plt.subplot(235)
    # plt.title("Noisy image (measurements)")
    # plt.imshow(y, cmap='gray',vmin=vmin, vmax=vmax)
    # plt.subplot(236)
    # plt.title(f"Reconstruction: sum")
    # plt.imshow(x1_cv+x2_cv, cmap='gray',vmin=vmin, vmax=vmax)
    # plt.suptitle(fr"$\lambda_1$: {lambda1:.3f}, $\lambda_2$: {lambda2:.3f}, Solved in {solving_time_cv:.2f}s")
    # plt.show()

    # print("Value fo the different terms of the objective function:")
    # print(f"\tData fidelity: {0.5 * np.linalg.norm(y - x1_cv - x2_cv)**2:.3e}")
    # print(f"\tSmooth penalty: {.5 * lambda2 * np.linalg.norm(lap(x2_cv.ravel()))**2:.3e}")
    # print(f"\tSparse penalty: {lambda1 * np.linalg.norm(l21(grad(x1_cv.ravel()))):.3e}")