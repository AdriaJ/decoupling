import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy import distance
import pyxu.operator as pxop
from ezgpx import GPX

import nullspace.d2shampoo as d2s

# todo idea:
#  - Remove points that are too close from each other

filename = "cham_zermatt.gpx"  #"chezine.gpx", "cham_zermatt.gpx"
downsample = 20
# Possibility to do smart downsampling: when two points are too close from each other for instance

lambda_factor = 1e-3
eps_pds = 1e-4
eps_apgd = 1e-6
thresh = 1e-4
manual_thresholding = False

if __name__ == "__main__":
    datadir = os.path.join("/home/jarret/PycharmProjects/decoupling", "data")
    # Load the GPX file
    gpx = GPX(os.path.join(datadir, filename))

    df = gpx.to_dataframe()

    print("Dataframe shape: ", df.shape)
    print("Dataframe columns: ", df.columns)

    steps = [0.]
    for i in range(1, df.shape[0]):
        steps.append(distance.distance((df["lat"][i - 1], df["lon"][i - 1]), (df["lat"][i], df["lon"][i])).meters)
    df["step"] = steps
    df["cumdist"] = df["step"].cumsum()

    # remove the first rows until movement
    i = 0
    while df["step"][i + 1] == 0:
        df.drop(i, inplace=True)
        i += 1
    #remove other data points without movement
    df.drop(df[df["step"] == 0].index[1:], inplace=True)

    # reset index
    df.reset_index(drop=True, inplace=True)

    print(f"Number of measurement points after downsample: {df.shape[0] // downsample}")

    x = df["cumdist"][::downsample].values
    y = df["ele"][::downsample].values
    # x = x[:x.shape[0] // 4]
    # y = y[:y.shape[0] // 4]

    Lop = d2s.Lop(x)
    print(r"Computation of $\lambda_{\max}$ ...")
    start = time.time()
    lambda_max = d2s.lambda_max(x, y, Lop, damp=1e-12)
    lmax_time = time.time() - start
    print(f"\tDone in: {lmax_time}s")
    # print(f"\tValue: {lambda_max:.2f}")

    # # Evalute lambda_max value and reconstruction time as a funciton of dampening factor:
    # damps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    # vals = []
    # times = []
    # for damp in damps:
    #     start = time.time()
    #     lambda_max = d2s.lambda_max(x, y, Lop, damp=damp)
    #     lmax_time = time.time() - start
    #     vals.append(lambda_max)
    #     times.append(lmax_time)
    #
    # plt.figure()
    # plt.xscale('log')
    # plt.scatter(damps, times, label="times")
    # plt.legend()
    # plt.twinx()
    # plt.scatter(damps, vals, label="values", marker='+', color='red')
    # plt.legend()
    # plt.show()
    #
    #  -> CONCLUSION: requires at least 1e-10 damp factor

    ##### Restart computations from here

    lambda_ = lambda_factor * lambda_max
    ylambda, t1 = d2s.compute_ylambda(x, y, Lop, lambda_, eps_pds)
    print(f"Computation of $y_\lambda$ converged: {d2s.compute_ylambda_converged(y, ylambda, lambda_, Lop, thresh=1e-5)}")
    # todo explore the certificate of optimality
    certif37 = Lop.T.pinv(y-ylambda, 1e-10)/lambda_
    Lylambda = Lop(ylambda)

    plt.figure()
    plt.scatter(np.arange(certif37.shape[0]), certif37, label='certif')
    plt.scatter(np.arange(Lylambda.shape[0]), Lylambda, label='fit')
    plt.hlines(1, 0, certif37.shape[0])
    plt.suptitle(r"Certificate for $y_\lambda$ estimation")
    plt.title("Check that certif is always smaller than 1 and saturates when fit is non zero.")
    plt.legend()
    plt.show()

    adeb = d2s.canonical_interpolant_coeffs(x, ylambda)
    ajar, t2 = d2s.decoupled_solving(x, y, lambda_, eps_apgd)

    if manual_thresholding:
        ajar[np.abs(ajar) < thresh] = 0
        adeb[np.abs(adeb) < thresh] = 0

    print(f"Time for Debarre method: {t1}")
    print(f"Time for Jarret method: {t2}")
    print(f"Identical solution? {np.allclose(adeb, ajar)}")
    print(f"Sparsity of the methods:")
    print(f"\tDebarre: {np.count_nonzero(adeb)}/{adeb.shape[0]}")
    print(f"\tJarret: {np.count_nonzero(ajar)}/{ajar.shape[0]}")

    solution = d2s.fcano(ajar, x)
    solution_deb = d2s.fcano(adeb, x)
    _, certif = d2s.canonical_certificate(x, ajar)

    certif37 = Lop.T.pinv(y-solution(x), 1e-10)/lambda_
    Lylambda = Lop(solution(x))

    plt.figure()
    plt.scatter(np.arange(certif37.shape[0]), certif37, label='certif')
    plt.scatter(np.arange(Lylambda.shape[0]), np.sign(Lylambda), label='fit')
    plt.hlines([1, -1], 0, certif37.shape[0])
    plt.suptitle(r"Certificate for $y_\lambda$ estimation with Jarret's solution")
    plt.title("Check that certif is always smaller than 1 and saturates when fit is non zero.")
    plt.legend()
    plt.show()

    df = .5 * pxop.SquaredL2Norm(y.shape[0]).asloss(y)
    cost_deb = df.apply(solution_deb(x)) + lambda_ * np.linalg.norm(adeb[1:-1], 1)
    cost_jar = df.apply(solution(x)) + lambda_ * np.linalg.norm(ajar[1:-1], 1)
    print(f"Value of the objective function:")
    print(f'\tCost jar: {cost_jar[0]:.3e}')
    print(f'\tCost deb: {cost_deb[0]:.3e}')

    plt.figure(figsize=(14, 14))
    ax1 = plt.subplot(211)
    plt.scatter(x, y, marker='+', s=20, color='skyblue', alpha=.5, label="Data points")
    # plt.scatter(df["cumdist"][::downsample], df["ele"][::downsample], marker='.', s=10, color='red')
    plt.scatter(x, ylambda, marker='+', s=20, color='green', label=r"$y_\lambda$")
    plt.plot(x, solution(x), color='black', label='Decoupled solution')
    plt.plot(x, solution_deb(x), color='orange', label='Solution Deb')
    plt.legend()
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(x, certif(x))
    plt.scatter(x, np.zeros_like(x), marker='+', color='k')
    plt.show()

    maxi = max(np.abs(ajar).max(), np.abs(adeb).max())
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    # plt.yscale('log')
    plt.hist(np.abs(ajar), bins=np.logspace(start=np.log10(1e-8), stop=np.log10(maxi), num=20))
    plt.xscale('log')
    plt.title("Jarret method")
    plt.subplot(122)
    plt.title("Debarre method")
    # plt.yscale('log')
    plt.hist(np.abs(adeb), bins=np.logspace(start=np.log10(1e-8), stop=np.log10(maxi), num=20))
    plt.xscale('log')
    plt.show()
