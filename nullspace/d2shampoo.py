import numpy as np
import matplotlib.pyplot as plt
import time

import pyxu.abc as pxa
import pyxu.operator as pxop
import pyxu.opt.solver as pxls

L = 10
supp = (0, 1)
yrange = (-1, 1)
psnr = 30

lambda_factor = .1

seed = 10

if __name__ == "__main__":
    rng = np.random.default_rng(seed)

    x = rng.uniform(*supp, L)  # to sort maybe
    y = rng.uniform(*yrange, L)
    sigma2 = 10 ** (-psnr / 10) * np.abs(y).max() ** 2
    n = rng.normal(0, np.sqrt(sigma2), L)
    y += n

    # plot samples
    plt.figure()
    plt.scatter(x, y, marker='x')
    plt.show()

    # define v and L from (37)
    v = 1 / (x[1:] - x[:-1])
    Lmat = (np.diag(np.append(v, 0)) - np.diag(np.append(v[:-1] + v[1:], 0), 1) + np.diag(v[1:], 2) )[:-2, :]
    Lop = pxa.LinOp.from_array(Lmat)
    Lop.lipschitz = Lop.estimate_lipschitz()

    # compute lambda max with (40)
    # option 1
    fOp = pxa.LinOp.from_array(np.hstack([np.ones((L, 1)), x[:, None]]))
    f = .5 * pxop.SquaredL2Norm(fOp.shape[0]).asloss(y) * fOp
    f.diff_lipschitz = f.estimate_diff_lipschitz()
    nlcg = pxls.NLCG(f, show_progress=False)
    nlcg.fit(x0=np.zeros(2))
    reglin = nlcg.solution()

    # option 2: explicit form (has been verified already, valid)
    a0 = (1 / L) * (y.sum() - x.sum() * (np.dot(x, y) - x.sum() * y.sum() / L) / (np.dot(x, x) - x.sum() ** 2 / L))
    a1 = (np.dot(x, y) - x.sum() * y.sum() / L) / (np.dot(x, x) - x.sum() ** 2 / L)

    tmp = a0 + a1 * x - y
    lambda_max = np.abs(Lop.T.pinv(tmp, 1e-5)).max()  # to double check what indeed happens for larger lambdas
    lambda_ = lambda_factor * lambda_max

    # Solve the problem with their method
    datafid = .5 * pxop.SquaredL2Norm(L).asloss(y)
    datafid.diff_lipschitz = datafid.estimate_diff_lipschitz()
    h = lambda_ * pxop.L1Norm(Lop.shape[0])
    cv = pxls.CV(f=datafid, h=h, K=Lop, show_progress=True)
    x0, z0 = np.zeros(L), np.zeros(Lop.shape[0])
    cv.fit(x0=x0, z0=z0)
    y_lambda = cv.solution()

    #todo need to do the interpolation step now



