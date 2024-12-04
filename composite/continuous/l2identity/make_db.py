"""
Simulate data and reconstruct.
"""

import subprocess
import os
import numpy as np

from composite.continuous.l2identity.run_pipeline import snrdb

r12s = [.5, .75, 1.,]  # [.5, 1.5]  # [.5, .75, 1., 1.5, 2.]
fgbgR = 10.
# fgbgRs = [10., 5., 2., 1.]
reps = 10
l1fs = [.2, .3, .4]  # [.2, ]  #[.2, .3, .4]
l2s = [1e-4, 1e-3, 1e-2, 1e-1]  # [1e-3]  # [1e-4, 1e-3, 1e-2, 1e-1]
lfs = [.1, 0.2, 0.3, .4]  # [.3, ]  # [.1, 0.2, 0.3, .4]

snrdb = 10

cwd = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity"

if __name__ == "__main__":

    db_path = os.path.join(cwd, "database")
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    # for fgbgR in fgbgRs:
    #     print(f"Running pipeline with fgbgR: {fgbgR}")
    #     fgbgR_path = os.path.join(db_path, f"fgbgR_{fgbgR}")
    #     if not os.path.exists(fgbgR_path):
    #         os.makedirs(fgbgR_path)

    for r12 in r12s:
        print(f"Running pipeline with r12: {r12:.2f}")
        r12_path = os.path.join(db_path, f"r12_{r12}")
        if not os.path.exists(r12_path):
            os.makedirs(r12_path)
        for _ in range(reps):
            seed = np.random.randint(1_000_000)
            print(f"Running pipeline with seed: {seed}")
            seed_path = os.path.join(r12_path, f"{seed}")
            if not os.path.exists(seed_path):
                os.makedirs(seed_path)

            # subprocess.run(['python', os.path.join(cwd, "simulate_db.py"),
            #                 "--seed", f"{seed:d}", "--fgbgR", str(fgbgR), "--snr", str(snrdb),
            #                 "--save", seed_path], check=True, text=True)
            subprocess.run(['python', os.path.join(cwd, "simulate_db.py"),
                            "--seed", f"{seed:d}", "--fgbgR", str(fgbgR), "--snr", str(snrdb),
                            "--save", seed_path, "--r12", str(r12)], check=True, text=True)

            # saves a list of npz files, some named as blasso_*.npz and some as composite_*_*.npz,
            # where * stands for the actual float value of the regularization parameters.
            # If one argument is missing the associated reconstruction is not performed.
            subprocess.run(['python', os.path.join(cwd, "reconstruct_db.py"),
                            "--l1f", *(str(u) for u in l1fs),
                            "--l2", *(str(u) for u in l2s),
                            "--lf", *(str(u) for u in lfs),
                            "--data_path", seed_path], check=True)
