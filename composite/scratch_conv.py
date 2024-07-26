import numpy as np
import matplotlib.pyplot as plt

N = 201
sigma1 = 5
sigma2 = np.sqrt(2) * sigma1

X = np.arange(N)

int1 = np.sqrt(2 * np.pi * sigma1**2)
gauss1 = np.exp(-0.5 * (X - N // 2) ** 2 / sigma1 ** 2)
gauss1 *= 1 / int1
int2 = np.sqrt(2 * np.pi * sigma2**2)
gauss2 = np.exp(-0.5 * (X - N // 2) ** 2 / sigma2 ** 2)
gauss2 *= 1 / int2


discrete_conv = np.convolve(gauss1, gauss2, mode='same')
continuous_conv = np.sqrt(2 * np.pi * sigma1**2 * sigma2**2/(sigma1**2 + sigma2**2)) * np.exp(-0.5 * (X - N // 2) ** 2 / (sigma1**2 + sigma2**2))/int1

ylim = max(discrete_conv.max(), continuous_conv.max(), gauss1.max(), gauss2.max())

plt.figure(figsize=(10, 10))
plt.subplot(311)
plt.ylim(top=1.05 * ylim)
plt.title(rf"$\sigma_1 =$ {sigma1}")
plt.scatter(X, gauss1, c='r')
plt.subplot(312)
plt.ylim(top=1.05 * ylim)
plt.title(rf"$\sigma_2 =$ {sigma2}")
plt.scatter(X, gauss2, c='b')
plt.subplot(313)
plt.ylim(top=1.05 * ylim)
plt.title("Convolution")
plt.scatter(X, discrete_conv, c='g')
plt.scatter(X, continuous_conv, c='k', marker='+', alpha=.5)
plt.show()

# print(gauss1.sum(), np.sqrt(2 * np.pi * sigma1**2))

plt.figure()
plt.scatter(X, gauss1, c='r')
plt.scatter(X, gauss2, c='b')
plt.show()

rng = np.random.default_rng()
# First order peaks
k = 2
x1 = np.zeros(N)
x1[rng.choice(N, k, replace=False)] = rng.uniform(1, 10, k)

# Second order peaks
k = 10
x2p = np.zeros(N)
x2p[rng.choice(N, k, replace=False)] = rng.uniform(1, 5, k)
x2 = np.convolve(x2p, gauss2, mode='same')

plt.figure()
plt.subplot(311)
plt.stem(X, x1)
plt.subplot(312)
plt.scatter(X, x2, c='b')
plt.stem(X, x2p, markerfmt='bo')
plt.subplot(313)
plt.scatter(X, np.convolve(x1 + x2, gauss1, mode="same"), c='g')
plt.show()