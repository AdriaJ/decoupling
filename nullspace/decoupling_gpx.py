import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy import distance
import pyxu.operator as pxop
import pyxu.abc as pxa
import pyxu.opt.stop as pxst
import pyxu.opt.solver as pxls
from ezgpx import GPX

import nullspace.d2shampoo as d2s


filename = "cham_zermatt.gpx"  #"chezine.gpx", "cham_zermatt.gpx"
downsample = 1
min_dist = 200

# Possibility to do smart downsampling: when two points are too close from each other for instance

lambda_factor = 1e-3
eps_apgd = 1e-5

manual_thresholding = True
thresh = 1e-3

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
    # df['cumdist'] /= 1_000

    # remove the first rows until movement
    i = 0
    while df["step"][i + 1] == 0:
        df.drop(i, inplace=True)
        i += 1
    #remove other data points without movement
    df.drop(df[df["step"] == 0].index[1:], inplace=True)
    d = df['step'].values
    curr_sum = 0
    tokeep = [0]
    for i in range(1, df.shape[0]):
        curr_sum += d[i]
        if curr_sum > min_dist:
            tokeep.append(i)
            curr_sum = 0
    df = df.iloc[tokeep]
    # reset index
    df.reset_index(drop=True, inplace=True)

    print(f"Number of measurement points after downsample: {df.shape[0] // downsample}")

    x = df["cumdist"][::downsample].values
    y = df["ele"][::downsample].values
    # x = x[:x.shape[0] // 4]
    # y = y[:y.shape[0] // 4]

    start = time.time()
    Lop = d2s.Lop(x) # try to make it matrix free ?
    print(f"Time to instantiate Lop: {time.time() - start}")
    # start = time.time()
    # Lopmf = d2s.Lop_matrixfree(x) # try to make it matrix free ?
    # print(f"Time to instantiate Lop matrix-free: {time.time() - start}")

    print(f"Shape of Lop: {Lop.shape}")
    print(r"Computation of $\lambda_{\max}$ ...")
    start = time.time()
    lambda_max = d2s.lambda_max(x, y, Lop, damp=1e-10)
    lmax_time = time.time() - start
    print(f"\tDone in: {lmax_time}s")
    # start = time.time()
    # lambda_max2 = d2s.lambda_max(x, y, Lopmf, damp=1e-10)
    # lmax_time2 = time.time() - start
    # print(f"\tDone in with mf op: {lmax_time2}s")

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
    # ajar, t2 = d2s.decoupled_solving(x, y, lambda_, eps_apgd)

    # Investigate the certificate:
    start = time.time()
    L = x.shape[0]
    H = np.hstack([x[:, None], np.ones((L, 1))])
    gramHm1 = np.linalg.inv(H.T @ H)
    P = np.eye(L) - H @ gramHm1 @ H.T
    A = np.maximum(x[:, None] - x[None, 1:-1], 0)  # use formula (20)
    forward = pxa.QuadraticFunc(shape=(1, L), Q=pxa.LinOp.from_array(P)).asloss(y) * pxa.LinOp.from_array(A)
    forward.diff_lipschitz = forward.estimate_diff_lipschitz(method='svd')
    g = lambda_ * pxop.L1Norm(L - 2)
    pgd = pxls.PGD(f=forward, g=g, show_progress=False, verbosity=1_000)
    stop_crit_pgd = pxst.RelError(
        eps=eps_apgd,
        var="x",
        f=None,
        norm=2,
        satisfy_all=True,
    )
    pgd.fit(x0=np.zeros(L - 2), stop_crit=stop_crit_pgd, track_objective=True)
    # a_pgd = pgd.solution()
    var, hist = pgd.stats()
    a_pgd = var['x']
    residuals = y - A @ a_pgd
    b = gramHm1 @ H.T @ residuals
    ajar = np.zeros(L)
    ajar[1:-1] = a_pgd
    ajar[0] = b[0]
    ajar[-1] = b[1]
    time_jar = time.time() - start

    # plt.figure()
    # plt.imshow(A, interpolation='none')
    # plt.colorbar()
    # plt.show()

    eta = A.T @ P @ (y - A @ ajar[1:-1]) / lambda_

    plt.figure()
    plt.scatter(np.arange(eta.shape[0]), eta, color='red')
    plt.hlines([-1, 1], 0, eta.shape[0], color='k')
    plt.title("Dual certificate of coefficients estimation problem")
    plt.show()
    print(f"Sup of the certificate: {np.abs(eta).max():.2f}")


    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(121)
    plt.scatter(hist['iteration'], hist['RelError[x]'], label='Stopping criterion', marker='.', s=10)
    plt.yscale('log')
    plt.legend()
    ax2 = plt.subplot(122, sharex=ax1)
    plt.scatter(hist['iteration'], hist['Memorize[objective_func]'], label='Objective function', marker='.', s=10)
    plt.yscale('log')
    plt.legend()
    plt.suptitle(f"Lambda factor: {lambda_factor:.2e}")
    plt.suptitle("Convergence of the solver")
    plt.show()


    plt.figure(figsize=(8, 7))
    plt.hist(np.abs(ajar), bins=np.logspace(start=np.log10(1e-8), stop=np.log10(np.abs(ajar).max()), num=50))
    plt.xscale('log')
    plt.title("Histogram of coefficients")
    plt.show()

    print(f"Sparsity before manual thresholding:{np.count_nonzero(ajar)}/{ajar.shape[0]}")
    if manual_thresholding:
        ajar[np.abs(ajar) < thresh] = 0
    print(f"Sparsity after manual thresholding:{np.count_nonzero(ajar)}/{ajar.shape[0]}")

    print(f"Time for Jarret method: {time_jar}")

    solution = d2s.fcano(ajar, x)
    _, certif = d2s.canonical_certificate(x, ajar)

    # certif37 = Lop.T.pinv(y-solution(x), 1e-10, kwargs_init={'verbosity': 5_000})/lambda_
    # Lylambda = Lop(solution(x))
    # Lylambda[np.abs(Lylambda) < 1e-8] = 0
    #
    # plt.figure()
    # plt.scatter(np.arange(certif37.shape[0]), certif37, label='certif')
    # plt.scatter(np.arange(certif37.shape[0])[np.abs(certif37)>=1.], certif37[np.abs(certif37)>=1.], label='certif', color='red')
    # plt.scatter(np.arange(Lylambda.shape[0]), np.sign(Lylambda), label='fit')
    # plt.hlines([1, -1], 0, certif37.shape[0])
    # plt.suptitle(r"Certificate for $y_\lambda$ estimation with Jarret's solution")
    # plt.title("Check that certif is always smaller than 1 and saturates when fit is non zero.")
    # plt.legend()
    # plt.show()

    dfid = .5 * pxop.SquaredL2Norm(y.shape[0]).asloss(y)
    cost_jar = dfid.apply(solution(x)) + lambda_ * np.linalg.norm(ajar[1:-1], 1)
    print(f"Value of the objective function:")
    print(f'\tCost jar: {cost_jar[0]:.3e}')

    plt.figure(figsize=(14, 14))
    ax1 = plt.subplot(211)
    plt.scatter(x, y, marker='+', s=20, color='skyblue', alpha=1., label="Data points")
    # plt.scatter(df["cumdist"][::downsample], df["ele"][::downsample], marker='.', s=10, color='red')
    plt.plot(x, solution(x), color='black', label='Decoupled solution')
    plt.twinx()
    plt.stem(x[1:-1], ajar[1:-1], label="Change of slope (= Innovations of the spline)")
    plt.legend()
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(x, certif(x))
    plt.scatter(x, np.zeros_like(x), marker='+', color='k')
    plt.suptitle(f"Lambda factor: {lambda_factor:.2e}")
    plt.show()

    # todo try solving with PFW: faster solve ?
    # todo also try with other GPX tracks (maybe bike ?)

