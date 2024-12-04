"""
Load the source image and the reconstruction, for various values of the regularizatino parameters.
Plot the reconstruction.
Maybe evaluate some metric.
Compare with LASSO reconstruction.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

# seed = ...
# composite = True
# blasso = False
# gaussian = True

data_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/pipeline/data"

def relL2Error(reco, source):
    return np.linalg.norm(reco - source) / np.linalg.norm(source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed', required=True)

    parser.add_argument('--l1f', type=float, help='Factor for lambda1 (composite)', default=[.1, ], nargs='*')
    parser.add_argument('--l2', type=float, help='Value of lambda2 (composite)', default=[1.], nargs='*')
    parser.add_argument('--lf', type=float, help='Factor for lambda (BLASSO)', default=[.1], nargs='*')

    parser.add_argument('--fgbgR', type=float, default=10.)
    parser.add_argument('--snr', type=float, default=20.)
    parser.add_argument('--r12', type=float, default=1.)

    parser.add_argument('--composite', action='store_true')
    parser.add_argument('--blasso', action='store_true')
    parser.add_argument('--merge', action='store_true', help="Use Gaussian convolution to merge the recovered peaks.")

    parser.add_argument('--save', action='store_true', help="Save the plots.")

    args = parser.parse_args()

    seed_path = os.path.join(data_path, str(args.seed))
    reco_path = os.path.join(seed_path, "reco")
    if args.save:
        save_path = os.path.join(seed_path, "plots")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Load source and measurements
    data = np.load(os.path.join(seed_path, "data.npz"))

    # Plot the source image
    plt.figure(figsize=(6, 14))
    plt.subplot(411)
    plt.stem(data["img"])
    plt.title("Foreground")
    plt.subplot(412)
    plt.title("Background")
    plt.stem(data["background"])
    plt.subplot(413)
    plt.title("Sum")
    plt.stem(data["img"] + data["background"])
    plt.subplot(414)
    plt.title("Measurements")
    plt.stem(data["measurements"])
    plt.suptitle(f"fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB")
    if args.save:
        plt.savefig(os.path.join(save_path, "data.png"))
    else:
        plt.show()

    # Load the reconstructions
    if args.composite:
        filenames_comp = []
        comp_recos = []
        for entry in os.scandir(reco_path):
            if entry.is_file() and entry.name.startswith("comp"):
                filenames_comp.append(entry.name)
        filenames_comp.sort()
        for filename in filenames_comp:
            comp_recos.append(np.load(os.path.join(reco_path, filename)))

        Nrows, Ncols = len(args.l2), len(args.l1f)

        # Plot the sparse component
        fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=True,
                                figsize=(4 * Ncols + 2 * (Ncols-1), 4 * Nrows + 2 * (Nrows-1)))
        i = 0
        for reco, ax in zip(comp_recos, axs.flat):
            ax.stem(reco["x1"])
            # ax.set_title(rf"$\lambda_2 = {reco['lambda2'][0]:.2e}, \lambda_1 = {reco['lambda1'][0]:.2e}$")
            ax.set(xlabel=rf"$\lambda_1 = {reco['lambda1'][0]:.2e}  (f: {args.l1f[i%len(args.l1f)]:.2f})$",
                   ylabel=rf"$\lambda_2 = {reco['lambda2'][0]:.2e}$")
            ax.label_outer()
            i += 1
        fig.suptitle(f"Foreground (fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB)")
        if args.save:
            plt.savefig(os.path.join(save_path, "foreground.png"))
        else:
            fig.show()

        # Plot the background component
        fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=True,
                                figsize=(4 * Ncols + 2 * (Ncols-1), 4 * Nrows + 2 * (Nrows-1)))
        i = 0
        for reco, ax in zip(comp_recos, axs.flat):
            ax.hlines(0, 0, reco["x2"].shape[0], color='black', alpha=.5)
            ax.plot(np.arange(reco["x2"].shape[0]), reco["x2"])
            # ax.set_title(rf"$\lambda_2 = {reco['lambda2'][0]:.2e}, \lambda_1 = {reco['lambda1'][0]:.2e}$")
            ax.set(xlabel=rf"$\lambda_1 = {reco['lambda1'][0]:.2e}  (f: {args.l1f[i%len(args.l1f)]:.2f})$",
                   ylabel=rf"$\lambda_2 = {reco['lambda2'][0]:.2e}$")
            ax.label_outer()
            i += 1
        fig.suptitle(f"Background (fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB)")
        if args.save:
            plt.savefig(os.path.join(save_path, "background.png"))
        else:
            fig.show()

        plt.figure()
        plt.plot(np.arange(data["background"].shape[0]), data["background"])
        plt.hlines(0, 0, data["background"].shape[0], color='black', alpha=.5)
        plt.suptitle("Background")
        if args.save:
            plt.savefig(os.path.join(save_path, "background_alone.png"))
        else:
            plt.show()

        # Plot the aggregated spikes
        if args.merge:
            repr_std = 1.5
            representation_kernel = 1 / (np.sqrt(2 * np.pi * repr_std ** 2)) * np.exp(
                -0.5 * np.arange(-3 * repr_std, 3 * repr_std + 1) ** 2 / repr_std ** 2)
            fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=True,
                                    figsize=(4 * Ncols + 2 * (Ncols - 1), 4 * Nrows + 2 * (Nrows - 1)))
            i = 0
            for reco, ax in zip(comp_recos, axs.flat):
                repr_reco = np.convolve(reco["x1"], representation_kernel, mode="same")
                ax.plot(np.arange(repr_reco.shape[0]), repr_reco, c='orange', marker='.')
                # ax.set_title(rf"$\lambda_2 = {reco['lambda2'][0]:.2e}, \lambda_1 = {reco['lambda1'][0]:.2e}$")
                ax.set(xlabel=rf"$\lambda_1 = {reco['lambda1'][0]:.2e}    (f: {args.l1f[i%len(args.l1f)]:.2f})$",
                       ylabel=rf"$\lambda_2 = {reco['lambda2'][0]:.2e}$")
                ax.label_outer()
                i += 1
            fig.suptitle(f"Merged foreground (fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB)")
            if args.save:
                plt.savefig(os.path.join(save_path, "foreground_merged.png"))
            else:
                fig.show()

            plt.figure()
            repr_source = np.convolve(data["img"], representation_kernel, mode="same")
            plt.plot(np.arange(repr_source.shape[0]), repr_source, c='orange', marker='.')
            fig.suptitle(f"Source foreground (fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB)")
            if args.save:
                plt.savefig(os.path.join(save_path, "foreground_merged_source.png"))
            else:
                plt.show()

        comp_times = np.array([reco['t'][0] for reco in comp_recos]).reshape(Nrows, Ncols)
        errors_comp = np.array([relL2Error(reco["x1"], data["img"]) for reco in comp_recos]).reshape(Nrows, Ncols)
        print(r"Composite reconstruction times : row = $\lambda_2$, column = $\lambda_1$")
        print(comp_times)

    if args.blasso:
        filenames_blasso = []
        blasso_recos = []
        for entry in os.scandir(reco_path):
            if entry.is_file() and entry.name.startswith("blasso"):
                filenames_blasso.append(entry.name)
        filenames_blasso.sort()
        for filename in filenames_blasso:
            blasso_recos.append(np.load(os.path.join(reco_path, filename)))

        Nrows = len(args.lf)
        fig, axs = plt.subplots(Nrows + 1, 1, sharex=True, sharey=True,
                                figsize=(6, 4 * Nrows + 2 * (Nrows-1)))
        axs[0].stem(data["img"])
        axs[0].set_title("Source image")
        i = 0
        for reco, ax in zip(blasso_recos, axs.flat[1:]):
            ax.stem(reco["x"])
            ax.set_title(rf"$\lambda = {reco['lambda_'][0]:.2e}    (f: {args.lf[i]:.2f})$")
            i += 1
        fig.suptitle(f"BLASSO foreground (fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB)")
        if args.save:
            plt.savefig(os.path.join(save_path, "blasso.png"))
        else:
            fig.show()

        if args.merge:
            repr_std = 1.5
            representation_kernel = 1 / (np.sqrt(2 * np.pi * repr_std ** 2)) * np.exp(
                -0.5 * np.arange(-3 * repr_std, 3 * repr_std + 1) ** 2 / repr_std ** 2)
            fig, axs = plt.subplots(Nrows + 1, 1, sharex=True, sharey=True,
                                    figsize=(6, 4 * Nrows + 2 * (Nrows - 1)))
            repr_source = np.convolve(data["img"], representation_kernel, mode="same")
            axs[0].plot(np.arange(repr_source.shape[0]), repr_source, c='orange', marker='.')
            axs[0].set_title("Source image")
            i = 0
            for reco, ax in zip(blasso_recos, axs.flat[1:]):
                repr_reco = np.convolve(reco["x"], representation_kernel, mode="same")
                ax.plot(np.arange(repr_reco.shape[0]), repr_reco, c='orange', marker='.')
                ax.set_title(rf"$\lambda = {reco['lambda_'][0]:.2e}    (f: {args.lf[i]:.2f})$")
                i += 1
            fig.suptitle(f"BLASSO convolved(fgbgR: {args.fgbgR:.1f}, $r_{12}$: {args.r12:.1f}, SNR: {args.snr:.1f} dB)")
            if args.save:
                plt.savefig(os.path.join(save_path, "blasso_merged.png"), dpi=300)
            else:
                fig.show()

        blasso_times = np.array([reco['t'][0] for reco in blasso_recos])
        errors_blasso = np.array([relL2Error(reco["x"], data["img"]) for reco in blasso_recos])
        print("BLASSO reconstruction times")
        print(args.lf)
        print(blasso_times)

    # todo : add metric evaluation
    # Make a table with reconstruciton time

    print(r"Composite relative L2 errors : row = $\lambda_2$, column = $\lambda_1$")
    print(errors_comp)
    print(r"BLASSO relative L2 errors")
    print(errors_blasso)

    print(filenames_comp)
    print(filenames_blasso)

    print("Pairs of parameters values:")
    print([[(f"{l2:.2e}, {l1:.2e}") for l1 in args.l1f] for l2 in args.l2])
    print([f"{l:.2e}" for l in args.lf])
