"""
gt_data.npz keys: "img", "background", "measurements", "noise_meas"
composite_f1_l2.npz keys: "x1", "x2", "t", "lambda1", "lambda2"
blasso_f.npz keys: "x", "t", "lambda_"
"""
import os
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from matplotlib import use
use("Qt5Agg")

db_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/database"
figures_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/pipeline/figures"

save_pdf = True

def relL2Error(reco, source):
    return np.linalg.norm(reco - source) / np.linalg.norm(source)

def relL1Error(reco, source):
    return np.linalg.norm(reco - source, 1) / np.linalg.norm(source, 1)

def repr(signal, repr_std=1.5):
    representation_kernel = 1 / (np.sqrt(2 * np.pi * repr_std ** 2)) * np.exp(
        -0.5 * np.arange(-3 * repr_std, 3 * repr_std + 1) ** 2 / repr_std ** 2)
    return np.convolve(signal, representation_kernel, mode='same')

def path_error(repr_std, ord=2):
    path_error = {}
    # browse the folders in db_path
    for dir in os.listdir(db_path):
        assert dir.startswith("fgbgR_")
        fgbgR_path = os.path.join(db_path, dir)
        for seedtxt in os.listdir(fgbgR_path):
            seed_path = os.path.join(fgbgR_path, seedtxt)
            composite_recos = [f for f in os.listdir(seed_path) if f.startswith("composite")]
            blasso_recos = [f for f in os.listdir(seed_path) if f.startswith("blasso")]
            # Load source and measurements
            gtdata = np.load(os.path.join(seed_path, "gt_data.npz"))
            repr_x1gt = repr(gtdata["img"], repr_std=repr_std)
            # Relative l2 error after convolution with representation kernel
            blasso_errors = []
            composite_errors = []
            for comp in composite_recos:
                compdata = np.load(os.path.join(seed_path, comp))
                if ord==2:
                    err = relL2Error(repr(compdata["x1"], repr_std=repr_std), repr_x1gt)
                elif ord==1:
                    err = relL1Error(repr(compdata["x1"], repr_std=repr_std), repr_x1gt)
                path_error[os.path.join(seed_path, comp)] = err
            for bl in blasso_recos:
                bldata = np.load(os.path.join(seed_path, bl))
                if ord==2:
                    err = relL2Error(repr(bldata["x"], repr_std=repr_std), repr_x1gt)
                elif ord==1:
                    err = relL1Error(repr(bldata["x"], repr_std=repr_std), repr_x1gt)
                path_error[os.path.join(seed_path, bl)] = err
    return path_error

if __name__ == "__main__":

    list_paths = []
    for dirpath, dirnames, filenames in os.walk(db_path):
        if dirpath == db_path:
            fgbgRs = dirnames
        if len(dirnames) == 0:
            list_paths += [os.path.join(dirpath, f) for f in filenames if f.endswith(".npz")]

    # df = pd.DataFrame(columns=["path", "fgbgR", "seed", "type", "l1f", "l2", "lf"])
    tmp = {"path": [], "fgbgR": [], "seed": [], "type": [], "l1f": [], "l2": [], "lf": []}
    for path in list_paths:
        l = path.split('/')
        fgbgR = float(l[-3][6:])
        seed = int(l[-2])
        if l[-1].startswith("gt"):
            type = "gt"
            l1f, l2, lf = None, None, None
        else:
            parts = l[-1].split("_")
            if parts[0] == "composite":
                type = "composite"
                l1f, l2 = float(parts[1]), float(parts[2][:-4])
                lf = None
            elif parts[0] == "blasso":
                type = "blasso"
                l1f, l2 = None, None
                lf = float(parts[1][:-4])
        tmp["path"].append(path)
        tmp["fgbgR"].append(fgbgR)
        tmp["seed"].append(seed)
        tmp["type"].append(type)
        tmp["l1f"].append(l1f)
        tmp["l2"].append(l2)
        tmp["lf"].append(lf)
    df = pd.DataFrame(tmp)

    # fill the dataframe with computation time and errors (2 new columns)
    times = []
    for row in df.itertuples():
        if row[4] == "blasso" or row[4] == "composite":
            data = np.load(row[1])
            times.append(data["t"][0])
        else:
            times.append(None)
    df["time"] = times

    df["RelErr_1.5"] = df["path"].map(path_error(1.5))
    df["RelErr_3.0"] = df["path"].map(path_error(3.0))
    df["RelL1Err_1.5"] = df["path"].map(path_error(1.5, ord=1))
    df["RelL1Err_3.0"] = df["path"].map(path_error(3.0, ord=1))

    # for each fgbgR and each seed, extract the minimum error,
    # compare if the minimum error is obtained with the same regularization parameter
    # Plot the minimum median and interquartile minimum error
    # Also reconstruction time for best case

    #select reconstructions
    df_recos = df[df["type"] != "gt"]
    # df_recos.groupby(['fgbgR', 'seed', 'type']).agg({'RelErr_1.5': 'min'})
    idx15 = df_recos.groupby(['fgbgR', 'seed', 'type'])["RelErr_1.5"].idxmin()
    best_15 = df.loc[idx15]
    idx30 = df_recos.groupby(['fgbgR', 'seed', 'type'])["RelErr_3.0"].idxmin()
    best_30 = df.loc[idx30]

    # best_15.groupby(['fgbgR', 'type'])['RelErr_1.5'].agg(['mean', 'std', 'median', 'min', 'max'])
    # best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_1.5'].agg(['median']).plot()
    # best_15.groupby(['fgbgR', 'type'])['RelErr_1.5'].quantile([0.25, 0.75])

    fig = plt.figure(figsize=(12, 4))
    axes = fig.subplots(1, 2, sharey=True)
    ax = axes[0]
    blasso = best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_1.5'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_1.5'].quantile(0.25),
                     best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_1.5'].quantile(0.75), alpha=0.2, color='blue')
    # valb = best_15[best_15['type'] == 'blasso'][["fgbgR", 'RelErr_1.5']].values
    # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    compo = best_15[best_15['type'] == 'composite'].groupby(['fgbgR'])['RelErr_1.5'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_15[best_15['type'] == 'composite'].groupby(['fgbgR'])['RelErr_1.5'].quantile(0.25),
                     best_15[best_15['type'] == 'composite'].groupby(['fgbgR'])['RelErr_1.5'].quantile(0.75), alpha=0.2, color='red')
    # valc = best_15[best_15['type'] == 'composite'][["fgbgR", 'RelErr_1.5']].values
    # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    ax.set_xlabel("Contrast")
    ax.set_title(r"Relative L2 error, $\sigma=1.5$")
    ax.legend()

    ax = axes[1]
    blasso = best_30[best_30['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_3.0'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_30[best_30['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_3.0'].quantile(0.25),
                     best_30[best_30['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_3.0'].quantile(0.75), alpha=0.2, color='blue')
    # valb = best_30[best_30['type'] == 'blasso'][["fgbgR", 'RelErr_3.0']].values
    # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    compo = best_30[best_30['type'] == 'composite'].groupby(['fgbgR'])['RelErr_3.0'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_30[best_30['type'] == 'composite'].groupby(['fgbgR'])['RelErr_3.0'].quantile(0.25),
                        best_30[best_30['type'] == 'composite'].groupby(['fgbgR'])['RelErr_3.0'].quantile(0.75), alpha=0.2, color='red')
    # valc = best_30[best_30['type'] == 'composite'][["fgbgR", 'RelErr_3.0']].values
    # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    ax.set_xlabel("Contrast")
    ax.set_title(r"Relative L2 error, $\sigma=3.0$")
    ax.legend()
    if save_pdf:
        plt.savefig(os.path.join(figures_path, "metrics_rl2.pdf"))
    plt.show()

    # select reconstructions for L1 error
    df_recos = df[df["type"] != "gt"]
    L1idx15 = df_recos.groupby(['fgbgR', 'seed', 'type'])["RelL1Err_1.5"].idxmin()
    L1best_15 = df.loc[L1idx15]
    L1idx30 = df_recos.groupby(['fgbgR', 'seed', 'type'])["RelL1Err_3.0"].idxmin()
    L1best_30 = df.loc[L1idx30]

    # Relative L1 error
    fig = plt.figure(figsize=(12, 4))
    axes = fig.subplots(1, 2, sharey=True)
    ax = axes[0]
    blasso = L1best_15[L1best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelL1Err_1.5'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, L1best_15[L1best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelL1Err_1.5'].quantile(0.25),
                     L1best_15[L1best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelL1Err_1.5'].quantile(0.75), alpha=0.2, color='blue')
    # valb = L1best_15[L1best_15['type'] == 'blasso'][["fgbgR", 'RelL1Err_1.5']].values
    # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    compo = L1best_15[L1best_15['type'] == 'composite'].groupby(['fgbgR'])['RelL1Err_1.5'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, L1best_15[L1best_15['type'] == 'composite'].groupby(['fgbgR'])['RelL1Err_1.5'].quantile(0.25),
                     L1best_15[L1best_15['type'] == 'composite'].groupby(['fgbgR'])['RelL1Err_1.5'].quantile(0.75), alpha=0.2, color='red')
    # valc = L1best_15[L1best_15['type'] == 'composite'][["fgbgR", 'RelL1Err_1.5']].values
    # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    ax.set_xlabel("Contrast")
    ax.set_title(r"Relative L1 error, $\sigma=1.5$")
    ax.legend()

    ax = axes[1]
    blasso = L1best_30[L1best_30['type'] == 'blasso'].groupby(['fgbgR'])['RelL1Err_3.0'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, L1best_30[L1best_30['type'] == 'blasso'].groupby(['fgbgR'])['RelL1Err_3.0'].quantile(0.25),
                     L1best_30[L1best_30['type'] == 'blasso'].groupby(['fgbgR'])['RelL1Err_3.0'].quantile(0.75), alpha=0.2, color='blue')
    # valb = L1best_30[L1best_30['type'] == 'blasso'][["fgbgR", 'RelL1Err_3.0']].values
    # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    compo = L1best_30[L1best_30['type'] == 'composite'].groupby(['fgbgR'])['RelL1Err_3.0'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, L1best_30[L1best_30['type'] == 'composite'].groupby(['fgbgR'])['RelL1Err_3.0'].quantile(0.25),
                        L1best_30[L1best_30['type'] == 'composite'].groupby(['fgbgR'])['RelL1Err_3.0'].quantile(0.75), alpha=0.2, color='red')
    # valc = L1best_30[L1best_30['type'] == 'composite'][["fgbgR", 'RelL1Err_3.0']].values
    # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    ax.set_xlabel("Contrast")
    ax.set_title(r"Relative L1 error, $\sigma=3.0$")
    ax.legend()
    if save_pdf:
        plt.savefig(os.path.join(figures_path, "metrics_l1.pdf"))
    plt.show()


    # Reconstruction time
    fig = plt.figure(figsize=(15, 7))
    axes = fig.subplots(1, 2, sharey=True)
    ax = axes[0]
    blasso = best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['time'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['time'].quantile(0.25),
                     best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['time'].quantile(0.75), alpha=0.2, color='blue')
    compo = best_15[best_15['type'] == 'composite'].groupby(['fgbgR'])['time'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_15[best_15['type'] == 'composite'].groupby(['fgbgR'])['time'].quantile(0.25),
                     best_15[best_15['type'] == 'composite'].groupby(['fgbgR'])['time'].quantile(0.75), alpha=0.2, color='red')
    ax.set_xlabel("Contrast")
    ax.set_title(r"Reconstruction time for best case with $\sigma=1.5$")
    ax.legend()

    ax = axes[1]
    blasso = best_30[best_30['type'] == 'blasso'].groupby(['fgbgR'])['time'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_30[best_30['type'] == 'blasso'].groupby(['fgbgR'])['time'].quantile(0.25),
                     best_30[best_30['type'] == 'blasso'].groupby(['fgbgR'])['time'].quantile(0.75), alpha=0.2, color='blue')
    compo = best_30[best_30['type'] == 'composite'].groupby(['fgbgR'])['time'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_30[best_30['type'] == 'composite'].groupby(['fgbgR'])['time'].quantile(0.25),
                        best_30[best_30['type'] == 'composite'].groupby(['fgbgR'])['time'].quantile(0.75), alpha=0.2, color='red')
    ax.set_xlabel("Contrast")
    ax.set_title(r"Reconstruction time for best case with $\sigma=3.0$")
    ax.legend()
    plt.show()


    fgbgRs = []
    res = {}
    # path_error = {}

    # browse the folders in db_path
    for dir in os.listdir(db_path):
        assert dir.startswith("fgbgR_")
        fgbgR = float(dir[6:])
        fgbgRs.append(fgbgR)
        fgbgR_path = os.path.join(db_path, dir)

        res_fgbgR = {}

        for seedtxt in os.listdir(fgbgR_path):
            seed = int(seedtxt)
            seed_path = os.path.join(fgbgR_path, seedtxt)

            composite_recos = [f for f in os.listdir(seed_path) if f.startswith("composite")]
            blasso_recos = [f for f in os.listdir(seed_path) if f.startswith("blasso")]

            # Load source and measurements
            gtdata = np.load(os.path.join(seed_path, "gt_data.npz"))
            repr_x1gt = repr(gtdata["img"])

            # Relative l2 error after convolution with representation kernel
            blasso_errors = []
            composite_errors = []
            for comp in composite_recos:
                compdata = np.load(os.path.join(seed_path, comp))
                err = relL2Error(repr(compdata["x1"]), repr_x1gt)
                # path_error[os.path.join(seed_path, comp)] = err
                composite_errors.append(err)
            for bl in blasso_recos:
                bldata = np.load(os.path.join(seed_path, bl))
                err = relL2Error(repr(bldata["x"]), repr_x1gt)
                # path_error[os.path.join(seed_path, bl)] = err
                blasso_errors.append(err)

            # minimum error and associated regularization parameters
            min_comp_err = min(composite_errors)
            min_comp_idx = composite_errors.index(min_comp_err)
            min_comp_name = composite_recos[min_comp_idx]
            l1f_min, l2_min = re.split("_", min_comp_name)[1:]
            l2_min = l2_min[:-4]

            min_bl_err = min(blasso_errors)
            min_bl_idx = blasso_errors.index(min_bl_err)
            min_bl_name = blasso_recos[min_bl_idx]
            lf_min = re.split("_", min_bl_name)[1]
            lf_min = lf_min[:-4]

            res_seed = {"comp_err": min_comp_err, "bl_err": min_bl_err,
                        "l1f": l1f_min, "l2": l2_min, "lf": lf_min}

            res_fgbgR[seedtxt] = res_seed

        res[dir] = res_fgbgR

    sorted_keys = list(res.keys())
    sorted_keys.sort(key=lambda x: float(x[6:]), reverse=True)
    print(sorted_keys)
    sorted_contrast = [float(k[6:]) for k in sorted_keys]

    # aggregate the results
    agg = {}
    items = ["bl_err", "lf", "comp_err", "l1f", "l2"]
    for item in items:
        agg[item] = np.array([[res[k][s][item] for s in res[k].keys()] for k in sorted_keys])


    # for k in res.keys():
    #     bl_errors = [res[k][s]["bl_err"] for s in res[k].keys()]
    #     best_lfs = [res[k][s]["lf"] for s in res[k].keys()]
    #     comp_errors = [res[k][s]["comp_err"] for s in res[k].keys()]
    #     best_lambdas = [(res[k][s]["l1f"], res[k][s]["l2"]) for s in res[k].keys()]
    #     agg[k] = {"bl_errors": bl_errors, "best_lfs": best_lfs,
    #               "comp_errors": comp_errors, "best_lambdas": best_lambdas}

    # print errors with respect to contrast
    plt.figure()
    plt.scatter(sorted_contrast, agg["bl_err"].mean(axis=1), label="BLASSO")
    plt.scatter(sorted_contrast, agg["comp_err"].mean(axis=1), label="Composite")
    #shade interquartile area
    plt.fill_between(sorted_contrast, np.percentile(agg["bl_err"], 25, axis=1), np.percentile(agg["bl_err"], 75, axis=1), alpha=0.2)
    plt.fill_between(sorted_contrast, np.percentile(agg["comp_err"], 25, axis=1), np.percentile(agg["comp_err"], 75, axis=1), alpha=0.2)
    plt.xlabel("Contrast")
    plt.ylabel("Relative L2 error")
    plt.legend()
    plt.show()

    # ----------------------------
    ## BLASSO error w.r.t lambda_f
    # get sorted lfs
    seed_dir = os.path.join(db_path, sorted_keys[0])
    valid_seed = os.listdir(seed_dir)[0]
    seed_dir = os.path.join(seed_dir, valid_seed)
    lfs_names = [f for f in os.listdir(seed_dir) if f.startswith("blasso")]
    sorted_lfs_names = sorted(lfs_names, key=lambda x: float(x[7:-4]))

    # for each fgbgR, extract a matrix of size nreps x len(lfs) with error
    # blasso error w.r.t. lfs
    bl_wrt_lfs = []

    for dir in sorted_keys:
        fgbgR_path = os.path.join(db_path, dir)
        tmp = []
        for seedtxt in os.listdir(fgbgR_path):
            seed_path = os.path.join(fgbgR_path, seedtxt)
            # Load source and measurements
            gtdata = np.load(os.path.join(seed_path, "gt_data.npz"))
            repr_x1gt = repr(gtdata["img"])
            tmp_seed = []
            for bl_name in sorted_lfs_names:
                bldata = np.load(os.path.join(seed_path, bl_name))
                err = relL2Error(repr(bldata["x"]), repr_x1gt)
                tmp_seed.append(err)
            tmp.append(tmp_seed)
        bl_wrt_lfs.append(np.array(tmp))

    sorted_lfs = [float(lf[7:-4]) for lf in sorted_lfs_names]
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.suptitle("BLASSO error w.r.t. $\lambda_f$")
    for i in range(4):
        ax = axes.flat[i]
        ax.set_title(f"fgbgR = {sorted_keys[i][6:]}")
        ax.scatter(sorted_lfs, bl_wrt_lfs[i].mean(axis=0))
        ax.fill_between(sorted_lfs, np.percentile(bl_wrt_lfs[i], 25, axis=0), np.percentile(bl_wrt_lfs[i], 75, axis=0), alpha=0.2)
    fig.show()
    # ----------------------------







