import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft

import pyxu.operator as pxop

N = 256

def laplace_primitive_noise(N, seed=None, mu=0, sigma=1):
    if seed is None:
        seed = np.random.randint(1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed=seed)

    gaussian = rng.normal(mu, sigma, (N, N))
    gaussian_freq = sfft.fft2(gaussian)

    frequency_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
                              np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2)
    frequency_response[0, 0] = 1.

    laplace_prim = sfft.ifft2(gaussian_freq / frequency_response).real

    return laplace_prim

def square_laplace_primitive_noise(N, seed=None, mu=0, sigma=1):
    if seed is None:
        seed = np.random.randint(1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed=seed)

    gaussian = rng.normal(mu, sigma, (N, N))
    gaussian_freq = sfft.fft2(gaussian)

    frequency_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
                              np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2)
    frequency_response[0, 0] = 1.

    laplace_prim = sfft.ifft2(gaussian_freq / frequency_response**2).real

    return laplace_prim

if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed=seed)

    gaussian = rng.normal(0, 1, (N, N))
    gaussian_freq = sfft.fft2(gaussian)

    plt.figure()
    plt.subplot(121)
    plt.imshow(gaussian, cmap='gray', interpolation='none')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(np.abs(np.fft.fftshift(gaussian_freq)), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

    lap = pxop.Laplacian((N, N), mode='wrap')
    d2gaussian = lap(gaussian.ravel()).reshape(N, N)
    d2gaussian_freq = sfft.fft2(d2gaussian)
    freqs = np.roll(np.arange(-N//2+1, N//2+1), N//2 + 1)
    grid_freq = (freqs[:, None]**2 + freqs[None, :]**2) / (N ** 2)
    # grid_freq[0, 0] = 1.
    candidate_freq_d2 = gaussian_freq * grid_freq

    plt.figure()
    plt.subplot(131)
    plt.imshow(d2gaussian, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.abs(np.fft.fftshift(d2gaussian_freq)), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(np.fft.fftshift(candidate_freq_d2)), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

    ### Generate some sort of second order primitive to gaussian noise

    grid_freqs_bis = grid_freq.copy()
    grid_freqs_bis[0, 0] = 1./N**2
    gaussian_prim = sfft.ifft2(sfft.fft2(gaussian) / grid_freqs_bis).real

    plt.figure()
    plt.subplot(131)
    plt.imshow(gaussian_prim, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.abs(sfft.fftshift(sfft.fft2(lap(gaussian_prim.ravel()).reshape((N, N))))), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(sfft.fftshift(gaussian_freq)), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

    test_one = np.zeros((N, N))
    test_one[0, 0] = 1
    kernel = lap(test_one.ravel()).reshape(N, N)
    f = sfft.fft2(kernel)
    print(np.allclose(f.imag, np.zeros_like(f)))
    freal = f.real
    # np.all(freal <= 0.)
    # np.sum(freal == 0.)
    # freal[0, 0]

    plt.figure()
    plt.imshow(sfft.fftshift(freal), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

    analytic_spectrum = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] + np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2)
    # np.allclose(analytic_spectrum, freal)

    analytic_spectrum[0, 0] = 1.
    laplace_prim = sfft.ifft2(gaussian_freq / analytic_spectrum).real
    plt.figure()
    plt.imshow(laplace_primitive_noise(N), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.imshow((lap(laplace_primitive_noise(N).ravel()).reshape((N, N))), cmap='gray', interpolation='none')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(gaussian, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

    reps = 50
    res = []
    for sigma in [1, 2, 4, 8]:
        res_s = []
        for _ in range(reps):
            res_s.append(laplace_primitive_noise(N, sigma=sigma).std())
        res.append(res_s)

    plt.figure()
    # plt.scatter([1, 2, 4, 8], np.mean(res, axis=1))
    plt.errorbar([1, 2, 4, 8], np.mean(res, axis=1), yerr=np.std(res, axis=1), fmt='o')
    plt.show()

    print(np.mean(res, axis=1)/np.array([1, 2, 4, 8]))

    plt.figure()
    for i, sigma in enumerate([1, 2, 4, 8]):
        plt.subplot(1, 4, i+1)
        plt.imshow(laplace_primitive_noise(N, sigma=sigma), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.title(fr"$\sigma = {sigma}$")
    plt.show()

    plt.figure()
    for i, sigma in enumerate([1, 2, 4, 8]):
        plt.subplot(1, 4, i+1)
        plt.imshow(square_laplace_primitive_noise(N, sigma=sigma), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.title(fr"$\sigma = {sigma}$")
    plt.show()
