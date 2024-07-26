import numpy as np
import matplotlib.pyplot as plt
import time

import pyxu.abc as pxa
import pyxu.operator as pxop
import pyxu.opt.solver as pxls
import pyxu.opt.stop as pxst

__all__ = ['canonical_interpolant_coeffs',
           'canonical_certificate',
           'fcano',
           'lambda_max',
           'compute_ylambda',
           'decoupled_solving',
           'Lop',
           'compute_ylambda_converged']

L = 10
supp = (0, 1)
yrange = (-1, 1)
psnr = 20
eps_pds = 1e-8
eps_apgd = 1e-6

lambda_factor = 1.2

seed = 102  # Uniqueness: 102

manual_thresholding = True
thresh = 1e-5


# Solution Debarre:
# 1. Compute ylambda
# 2. Compute the canonical interpolant coefficients
# 3. (Optional) Threshold the coefficients
# Solution Jarret:
# Directly compute the canonical interpolant coefficients with decoupled problems
# 1. Compute the innovations
# 2. Deduce the nullspace component coefficients

def Lop(x):
    """
    Lop = Finite diff @ Diagonal op @ Finite diff ?
    Finite diff: y[n] = x[n] - x[n-1]
    Parameters
    ----------
    x

    Returns
    -------

    """
    v = 1 / (x[1:] - x[:-1])
    Lmat = (np.diag(np.append(v, 0)) - np.diag(np.append(v[:-1] + v[1:], 0), 1) + np.diag(v[1:], 2))[:-2, :]
    op = pxa.LinOp.from_array(Lmat)
    op.lipschitz = op.estimate_lipschitz(method='svd')
    return op

def Lop_matrixfree(x):
    # test Lop
    L = x.shape[0]
    gradL = pxop.SubSample(L, slice(0, L-1)) * pxop.Gradient((L,), diff_method='fd')
    gradLm1 = pxop.SubSample(L-1, slice(0, L-2)) * pxop.Gradient((L-1,), diff_method='fd')
    v = 1 / (x[1:] - x[:-1])
    op = gradLm1 * pxop.DiagonalOp(v) * gradL
    op.lipschitz = op.estimate_lipschitz(method='svd')
    return op


def lambda_max(x, y, Lop, damp=1e-10):
    """

    Parameters
    ----------
    x
    y
    Lop: pxop.LinOp
        Operator defined at equation (37)

    Returns
    -------

    """
    L = x.shape[0]
    a0 = (1 / L) * (y.sum() - x.sum() * (np.dot(x, y) - x.sum() * y.sum() / L) / (np.dot(x, x) - x.sum() ** 2 / L))
    a1 = (np.dot(x, y) - x.sum() * y.sum() / L) / (np.dot(x, x) - x.sum() ** 2 / L)
    tmp = a0 + a1 * x - y
    return np.abs(Lop.T.pinv(tmp, damp, kwargs_init={'verbosity': 5_000, 'show_progress': False})).max()


def compute_ylambda(x, y, Lop, lambda_, eps_pds):
    L = x.shape[0]
    start = time.time()
    datafid = .5 * pxop.SquaredL2Norm(L).asloss(y)
    datafid.diff_lipschitz = datafid.estimate_diff_lipschitz(method='svd')
    h = lambda_ * pxop.L1Norm(Lop.shape[0])
    cv = pxls.CV(f=datafid, h=h, K=Lop, show_progress=False, verbosity=1_000)
    stop_crit_x = pxst.RelError(
        eps=eps_pds,
        var="x",
        f=None,
        norm=2,
        satisfy_all=True,
    )
    stop_crit_z = pxst.RelError(
        eps=eps_pds,
        var="z",
        f=None,
        norm=2,
        satisfy_all=True,
    )
    x0, z0 = np.zeros(L), np.zeros(Lop.shape[0])
    cv.fit(x0=x0, z0=z0, stop_crit=stop_crit_x & stop_crit_z)
    y_lambda = cv.solution()
    time_deb = time.time() - start
    return y_lambda, time_deb

def compute_ylambda_converged(y, ylambda, lambda_, Lop, thresh=1e-5):
    Lmat = Lop.mat
    Lcogram = Lmat @ Lmat.T
    Lcogram_inv = np.linalg.inv(Lcogram)
    certif37 = (1 / lambda_) * Lcogram_inv @ Lmat @ (y - ylambda)
    Ltosol = Lmat @ ylambda
    suppLtosol = np.abs(Ltosol) > thresh
    return np.all(certif37[suppLtosol] == np.sign(Ltosol[suppLtosol])) and np.abs(certif37[~suppLtosol]).max() < 1.


def canonical_interpolant_coeffs(x0, y0):
    """
    Returns the canonical interpolant of the data (x0, y0)

    Parameters
    ----------
    x0: sampling points
    y0: sampled values
    """
    L = x0.shape[0]
    diffs = (y0[1:] - y0[:-1]) / (x0[1:] - x0[:-1])
    a = np.zeros(L)
    a[0] = diffs[0]
    a[1:-1] = (diffs[1:] - diffs[:-1])
    a[-1] = y0[0] - a[0] * x0[0]
    return a

    # compute canonical dual certificate


def fcano(a, x):
    return lambda u: a[0] * u + a[-1] + np.sum(a[1:-1, None] * np.maximum(0, np.r_[u][None, :] - x[1:-1, None]), axis=0)


def canonical_certificate(x, a):
    L = x.shape[0]
    H = np.vstack([np.maximum(0, x[None, :] - x[:-1, None]), np.ones(L), ])  # x])
    # print(H.shape, np.linalg.matrix_rank(H))
    # print("Condition number of H: ", np.linalg.cond(H))
    u = np.sign(a)
    u[0] = 0
    u[-1] = 0
    c = np.linalg.solve(H, u)
    # # check if <c, 1> = 0 and <c, x> = 0
    # print(np.dot(c, np.ones(L)))
    # print(np.dot(c, x))
    return c, lambda u: (c[None, :] * np.maximum(0, x[None, :] - np.r_[u][:, None])).sum(axis=1)


def decoupled_solving(x, y, lambda_, eps_apgd, ):
    L = x.shape[0]
    start = time.time()
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
    pgd.fit(x0=np.zeros(L - 2), stop_crit=stop_crit_pgd)
    a_pgd = pgd.solution()
    residuals = y - A @ a_pgd
    b = gramHm1 @ H.T @ residuals
    ajar = np.zeros(L)
    ajar[1:-1] = a_pgd
    ajar[0] = b[0]
    ajar[-1] = b[1]
    time_jar = time.time() - start
    return ajar, time_jar


if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
        print(f'Seed: {seed}')
    rng = np.random.default_rng(seed)

    # x = rng.uniform(*supp, L)
    # x.sort()
    x = np.linspace(supp[0] + 0.05, supp[1] - 0.05, L, endpoint=True)
    noiselessy = rng.uniform(*yrange, L)
    sigma2 = (10 ** (-psnr / 10)) * (np.abs(noiselessy).max() ** 2)
    n = rng.normal(0, np.sqrt(sigma2), L)
    y = noiselessy + n

    # # plot samples
    # plt.figure()
    # plt.scatter(x, y, marker='x')
    # plt.show()

    # # define v and L from (37)
    # v = 1 / (x[1:] - x[:-1])
    # Lmat = (np.diag(np.append(v, 0)) - np.diag(np.append(v[:-1] + v[1:], 0), 1) + np.diag(v[1:], 2))[:-2, :]
    # Lop = pxa.LinOp.from_array(Lmat)
    # Lop.lipschitz = Lop.estimate_lipschitz()

    Loperator = Lop(x)

    # # compute lambda max with (40)
    # # option 1
    # start = time.time()
    # fOp = pxa.LinOp.from_array(np.hstack([np.ones((L, 1)), x[:, None]]))
    # f = .5 * pxop.SquaredL2Norm(fOp.shape[0]).asloss(y) * fOp
    # f.diff_lipschitz = f.estimate_diff_lipschitz()
    # nlcg = pxls.NLCG(f, show_progress=False)
    # nlcg.fit(x0=np.zeros(2))
    # reglin = nlcg.solution()
    # lmax_time1 = time.time() - start
    #
    # # option 2: explicit form (has been verified already, valid)
    # start = time.time()
    # a0 = (1 / L) * (y.sum() - x.sum() * (np.dot(x, y) - x.sum() * y.sum() / L) / (np.dot(x, x) - x.sum() ** 2 / L))
    # a1 = (np.dot(x, y) - x.sum() * y.sum() / L) / (np.dot(x, x) - x.sum() ** 2 / L)
    # lmax_time2 = time.time() - start

    # print(f"Lambda max computation time (option 1): {lmax_time1}")
    # print(f"Lambda max computation time (option 2): {lmax_time2}")
    # print(reglin)
    # print(a0, a1)

    # tmp = a0 + a1 * x - y
    # lmax = np.abs(Lop.T.pinv(tmp, 1e-5)).max()  # todo: double check what indeed happens for larger lambdas
    lmax = lambda_max(x, y, Loperator, damp=1e-12)
    lambda_ = lambda_factor * lmax

    # # Solve the problem with their method
    # start = time.time()
    # datafid = .5 * pxop.SquaredL2Norm(L).asloss(y)
    # datafid.diff_lipschitz = datafid.estimate_diff_lipschitz()
    # h = lambda_ * pxop.L1Norm(Lop.shape[0])
    # cv = pxls.CV(f=datafid, h=h, K=Lop, show_progress=False)
    # stop_crit_x = pxst.RelError(
    #     eps=eps_pds,
    #     var="x",
    #     f=None,
    #     norm=2,
    #     satisfy_all=True,
    # )
    # stop_crit_z = pxst.RelError(
    #     eps=eps_pds,
    #     var="z",
    #     f=None,
    #     norm=2,
    #     satisfy_all=True,
    # )
    # x0, z0 = np.zeros(L), np.zeros(Lop.shape[0])
    # cv.fit(x0=x0, z0=z0, stop_crit=stop_crit_x & stop_crit_z)
    # y_lambda = cv.solution()
    # time_deb = time.time() - start

    ylambda, t1 = compute_ylambda(x, y, Loperator, lambda_, eps_pds)


    # Assert convergence of the estimation of ylambda
    print(f"Convergence of ylambda: {compute_ylambda_converged(y, ylambda, lambda_, Loperator, thresh=1e-5)}")

    adeb = canonical_interpolant_coeffs(x, ylambda)
    if manual_thresholding:
        adeb[np.abs(adeb) < 1e-5] = 0
    sol_deb = fcano(adeb, x)

    # # Solve the problem with nullspace decoupling
    # # The matrix H corresponds to the sampling of the elements of the nullspace of D2
    # H = np.hstack([x[:, None], np.ones((L, 1))])
    # gramHm1 = np.linalg.inv(H.T @ H)
    # P = np.eye(L) - H @ gramHm1 @ H.T
    # A = np.maximum(x[:, None] - x[None, 1:-1], 0)  # use formula (20)
    # start = time.time()
    # forward = pxa.QuadraticFunc(shape=(1, L), Q=pxa.LinOp.from_array(P)).asloss(y) * pxa.LinOp.from_array(A)
    # forward.diff_lipschitz = forward.estimate_diff_lipschitz()
    # g = lambda_ * pxop.L1Norm(L-2)
    # pgd = pxls.PGD(f=forward, g=g, show_progress=False)
    # stop_crit_pgd = pxst.RelError(
    #     eps=eps_apgd,
    #     var="x",
    #     f=None,
    #     norm=2,
    #     satisfy_all=True,
    # )
    # pgd.fit(x0=np.zeros(L-2), stop_crit=stop_crit_pgd)
    # a_pgd = pgd.solution()
    # time_jar = time.time() - start
    # residuals = y - A @ a_pgd
    # b = gramHm1 @ H.T @ residuals
    # ajar = np.zeros(L)
    # ajar[1:-1] = a_pgd
    # ajar[0] = b[0]
    # ajar[-1] = b[1]

    ajar, t2 = decoupled_solving(x, y, lambda_, eps_apgd)

    if manual_thresholding:
        ajar[np.abs(ajar) < 1e-5] = 0

    sol_jar = fcano(ajar, x)

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.scatter(x, noiselessy, label='noiselessy', marker='x')
    plt.scatter(x, y, label='y', marker='x')
    plt.scatter(x, ylambda, label='y_lambda', marker='x')
    plt.plot(np.linspace(*supp, 1000), sol_deb(np.linspace(*supp, 1000)), label='sol_deb')
    plt.plot(np.linspace(*supp, 1000), sol_jar(np.linspace(*supp, 1000)), label='sol_jar')
    plt.hlines(0, *supp, 'r', '--', alpha=.5, zorder=-1)
    plt.legend()
    plt.title('Canonical interpolant')
    # plt.show()

    # plot the dual certificates
    cdeb, certif_deb = canonical_certificate(x, adeb)
    cjar, certif_jar = canonical_certificate(x, ajar)
    # plt.figure()
    plt.subplot(122)
    plt.plot(np.linspace(supp[0] - .1, supp[1] + .1, 1000), certif_deb(np.linspace(supp[0] - .1, supp[1] + .1, 1000)),
             label='sol_deb')
    plt.plot(np.linspace(supp[0] - .1, supp[1] + .1, 1000), certif_jar(np.linspace(supp[0] - .1, supp[1] + .1, 1000)),
             label='sol_jar')
    plt.hlines(0, *supp, 'r', '--', alpha=.5, zorder=-1, color='k')
    plt.scatter(x, 0 * x, marker='x', color='r')
    plt.legend()
    plt.title('Canonical dual certificate')
    plt.show()

    # A posteriori evaluation of the cost functionals

    df = .5 * pxop.SquaredL2Norm(L).asloss(y)
    cost_deb = df.apply(sol_deb(x)) + lambda_ * np.linalg.norm(adeb[1:-1], 1)
    cost_jar = df.apply(sol_jar(x)) + lambda_ * np.linalg.norm(ajar[1:-1], 1)
    print(f'Cost deb: {cost_deb}')
    print(f'Cost jar: {cost_jar}')
    print(f'Time deb: {t1}')
    print(f'Time jar: {t2}')
    print(
        f"Relative difference between the spline coefficients: {np.linalg.norm(adeb - ajar, 2) / np.linalg.norm(adeb, 2):.2e}")

    # # test Lop
    # gradL = pxop.SubSample(L, slice(0, L-1)) * pxop.Gradient((L,), diff_method='fd')
    # gradLm1 = pxop.SubSample(L-1, slice(0, L-2)) * pxop.Gradient((L-1,), diff_method='fd')
    # v = 1 / (x[1:] - x[:-1])
    # testop = gradLm1 * pxop.DiagonalOp(v) * gradL
    # np.allclose(Loperator.mat, testop.asarray())


