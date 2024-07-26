"""
Load the simulated measurements.
Perform either composite or BLASSO reconstruction, potentially with many different regularization parameters.
Save the result to disk.
"""
import os
import time
import argparse

import numpy as np
import pyxu.operator as pxop
import pyxu.opt.solver as pxls

import scipy.fft as sfft

from pyxu.opt.stop import RelError, MaxIter
from pyxu.abc import QuadraticFunc

# lambda1_factors = ...
# lambda2s = ...
# lambda_factors = ...
# method = ...  # "composite" or "blasso"

data_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/pipeline/data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed', required=True)

    # potentially provide a list of regularization parameters
    parser.add_argument('--l1f', type=float, help='Factor for lambda1 (composite)', default=[.1, ], nargs='*')
    parser.add_argument('--l2', type=float, help='Value of lambda2 (composite)', default=[1.], nargs='*')
    parser.add_argument('--lf', type=float, help='Factor for lambda (BLASSO)', default=[.1], nargs='*')

    parser.add_argument('--method', type=str, help='Method', default='composite', choices=['composite', 'blasso'])
    parser.add_argument('--eps', type=float, help='Epsilon for stopping criterion', default=1e-5)

    args = parser.parse_args()

    exp_path = os.path.join(data_path, str(args.seed))
    save_path = os.path.join(exp_path, "reco")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the measurements
    data = np.load(os.path.join(exp_path, "data.npz"))
    img, background = data["img"], data["background"]
    measurements = data["measurements"]

    # Extract parameters
    Ngrid = img.shape[0]
    Nmeas = measurements.shape[0]
    ds_factor = Ngrid // Nmeas

    # Common operators
    # todo: maybe to be read later from config file
    kernel_std = 5  # Gaussian kernel std
    kernel_width = 3 * 2 * kernel_std + 1
    norm_meas = (np.sqrt(2 * np.pi) * kernel_std)
    kernel_measurement = np.exp(-0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / (kernel_std ** 2))
    kernel_measurement /= norm_meas

    # Forward operator on the fine grid
    fOp = pxop.Convolve(
        arg_shape=img.shape,
        kernel=[kernel_measurement,],
        center=[kernel_width // 2,],
        mode="wrap",  # constant
        enable_warnings=True,)
    fOp.lipschitz = fOp.estimate_lipschitz(method='svd', tol=1e-3)

    # Measurement operator
    ss = pxop.SubSample(Ngrid, slice(ds_factor//2, Ngrid, ds_factor))
    Hop = ss * fOp

    stop_crit = RelError(eps=args.eps, var="x", f=None, norm=2, satisfy_all=True, ) & MaxIter(10)

    if args.method == "composite":
        # Regul operator
        regul_std2 = 2 * kernel_std ** 2
        norm_regul = np.sqrt(2 * np.pi * regul_std2)
        diffs = np.arange(0, 4 * np.sqrt(regul_std2), ds_factor)
        diffs = np.hstack([-diffs[1:][::-1], diffs])
        kernel_regul = np.exp(-0.5 * diffs ** 2 / regul_std2)
        kernel_regul /= norm_regul

        for i, lambda2 in enumerate(args.l2):
            # usage of lambda 2
            M_kernel = kernel_regul / lambda2
            M_kernel[M_kernel.shape[0] // 2] += 1

            regul_width = kernel_regul.shape[0]
            h = np.zeros(Nmeas)
            h[:regul_width] = M_kernel
            h = np.roll(h, -regul_width // 2 + 1)
            hm1 = sfft.irfft(1 / sfft.rfft(h))
            MlambdaInv = pxop.Convolve(
                arg_shape=Nmeas,
                kernel=[hm1, ],
                center=[0, ],
                mode="wrap",  # constant
                enable_warnings=True,)
            MlambdaInv.lipschitz = MlambdaInv.estimate_lipschitz(method="svd", tol=1e-4)

            lambda1max = np.abs(Hop.adjoint(MlambdaInv(measurements).ravel())).max()
            for j, l1f in enumerate(args.l1f):
                lambda1 = l1f * lambda1max

                loss = QuadraticFunc((1, Nmeas), Q=MlambdaInv).asloss(measurements.ravel()) * Hop

                # Solve the problem
                regul = lambda1 * pxop.PositiveL1Norm(Ngrid)

                print("Solving composite minimization...")
                pgd = pxls.PGD(loss, g=regul, show_progress=False)
                start = time.time()
                pgd.fit(x0=np.zeros(img.size), stop_crit=stop_crit)
                pgd_time = time.time() - start

                x1 = pgd.solution()

                Mresiduals = MlambdaInv(measurements - Hop(x1))
                tmp = np.zeros(Ngrid)
                tmp[ds_factor // 2::ds_factor] = Mresiduals
                x2 = np.convolve(tmp, kernel_measurement, mode='same') / lambda2

                filename = f"comp_lambda2_{i}_lambda1f_{j}"
                np.savez(os.path.join(save_path, filename),
                         x1=x1, x2=x2,
                         t=np.r_[pgd_time], lambda1=np.r_[lambda1], lambda2=np.r_[lambda2])

            tk_res = MlambdaInv(measurements)
            tmp_tk = np.zeros(Ngrid)
            tmp_tk[ds_factor // 2::ds_factor] = tk_res
            tk_solution = np.convolve(tmp_tk, kernel_measurement, mode='same')
            np.savez(os.path.join(save_path, f"tk_lambda2_{i}"), tk_solution=tk_solution, lambda2=np.r_[lambda2])

    elif args.method == "blasso":
        lambda_max = np.abs(Hop.adjoint(measurements).ravel()).max()
        for i, lf in enumerate(args.lf):
            lambda_ = lf * lambda_max

            loss = pxop.SquaredL2Norm(Nmeas).asloss(measurements) * Hop
            regul = lambda_ * pxop.PositiveL1Norm(Ngrid)

            print("Solving BLASSO...")
            pgd = pxls.PGD(loss, g=regul, show_progress=False)
            start = time.time()
            pgd.fit(x0=np.zeros(img.size), stop_crit=stop_crit)
            pgd_time = time.time() - start

            x = pgd.solution()

            filename = f"blasso_lambdaf_{i}"
            np.savez(os.path.join(save_path, filename), x=x,
                     t=np.r_[pgd_time], lambda_=np.r_[lambda_])
