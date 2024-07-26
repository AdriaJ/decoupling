import numpy as np
import scipy.signal as sig
import pyxu.operator as pxop
import scipy.fft as sfft

seed = None

N = 100
downsample = 4

l2 = 1.

kernel_std_A = 2  # Gaussian kernel std
kernel_widthA = 2 * 3 * kernel_std_A + 1  # Length of the Gaussian kernel
kernel_std_L = 4  # Gaussian kernel std
kernel_widthL = 2 * 3 * kernel_std_L + 1  # Length of the Gaussian kernel


if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)

    rng = np.random.default_rng(seed=seed)

    kernel_A = np.exp(-0.5 * ((np.arange(kernel_widthA) - (kernel_widthA - 1) / 2) ** 2) / (kernel_std_A ** 2))
    freqA = sfft.rfftn(np.roll(np.pad(kernel_A, (0, N - kernel_widthA)), -kernel_widthA // 2 + 1))
    kernel_L = np.exp(-0.5 * ((np.arange(kernel_widthL) - (kernel_widthL - 1) / 2) ** 2) / (kernel_std_L ** 2))  # (1 / (2 * np.pi * kernel_std_L ** 2)) *
    freqL = sfft.rfftn(np.roll(np.pad(kernel_L, (0, N - kernel_widthL)), -kernel_widthL // 2 + 1))

    ss = pxop.SubSample(N, slice(0, None, downsample))

    x = rng.random(N//4)
    #sanity
    np.allclose(x, sfft.irfftn(sfft.rfftn(x), s=x.shape))

    # Right Hand Side
    rhsr = freqA * sfft.rfftn(ss.adjoint(x))  # l2 * freqL**2 *
    tmpl = freqA * rhsr
    tmp = ss.adjoint(ss(sfft.irfftn(tmpl, s=(N,))))
    rhs = sfft.irfftn(freqA * sfft.rfftn(tmp) + l2 * (freqL**2) * rhsr, s=(N,))

    # unit test convolution
    manual_conv = np.convolve(np.pad(ss.adjoint(x), (3 * kernel_std_A, 3 * kernel_std_A), mode='wrap'), kernel_A, mode='valid')
    print(np.allclose(manual_conv, sfft.irfftn(rhsr, (N,))))
    l2atx = np.convolve(np.pad(manual_conv, (3 * kernel_std_L, 3 * kernel_std_L), mode='wrap'), kernel_L, mode='valid')
    l2l2atx = np.convolve(np.pad(l2atx, (3 * kernel_std_L, 3 * kernel_std_L), mode='wrap'), kernel_L, mode='valid')
    aatx = ss(np.convolve(np.pad(manual_conv, (3 * kernel_std_A, 3 * kernel_std_A), mode='wrap'), kernel_A, mode='valid'))
    ataatx = np.convolve(np.pad(ss.adjoint(aatx), (3 * kernel_std_A, 3 * kernel_std_A), mode='wrap'), kernel_A, mode='valid')
    res = ataatx + l2 * l2l2atx
    print(np.allclose(res, rhs))
    print(np.allclose(l2l2atx, sfft.irfftn(freqL**2 * rhsr, s=(N,))))
    print(np.allclose(ataatx, sfft.irfftn(freqA * sfft.rfftn(tmp), s=(N,))))


    # Left Hand Side
    #pad, roll and subsample kernel of A, then go to the Fourier domain
    kernel_Ass = ss(np.roll(np.pad(np.convolve(kernel_A, kernel_A, mode='full'),
                                   (0, N - kernel_widthA*2 + 1)), -kernel_widthA + 1))
    freqAss = sfft.rfftn(kernel_Ass)

    Lambda2x = ss(sfft.irfftn((freqA**2) * (freqL**2) * sfft.rfftn(ss.adjoint(x)), s=(N,)))
    Lambda2x = sfft.irfftn(sfft.rfftn(Lambda2x)/freqAss, s=x.shape)
    tmp = ss(sfft.irfftn((freqA ** 2) * sfft.rfftn(ss.adjoint(x)), s=(N,)))
    lhs = sfft.irfftn(freqA * sfft.rfftn(ss.adjoint(tmp + l2 * Lambda2x)), s=(N,))
    print(np.allclose(ataatx, sfft.irfftn(freqA * sfft.rfftn(ss.adjoint(tmp), s=(N,)))))

    print(np.allclose(rhs, lhs))
    print(rhs-lhs)

    # actually, we only need to verify that A^T Lambda_2 = L_2^T L_2 A^T
    print(np.allclose(sfft.irfftn(freqA * sfft.rfftn(ss.adjoint(Lambda2x)), s=(N,)),
                      l2l2atx))

    print(sfft.irfftn(freqA * sfft.rfftn(ss.adjoint(Lambda2x)), s=(N,))[:10])
    print(l2l2atx[:10])