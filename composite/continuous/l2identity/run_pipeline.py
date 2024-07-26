"""
Run the simulation and reconstruciton pipeline.
"""
import subprocess
import os
import numpy as np

seed = None

fgbgR = 2
snrdb = 0

l1fs = [0.05, .1, .2, .3]  # [0.05, .1, 0.2]
l2s = [1e-4, 1e-3, 1e-2, 1e-1]  # [.1, 1., 10.]
lfs = [0.05, .1, 0.2, 0.3]

pipeline_path = "/home/jarret/PycharmProjects/decoupling/composite/continuous/l2identity/pipeline"

def run_script(script_path, *args):
    try:
        result = subprocess.run(['python', script_path] + list(args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        print(f"Script output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Script error:\n{e.stderr}")

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    print(f"Running pipeline with seed: {seed}")

    # optional arguments: '--fgbgR 10. --snr 20.'
    run_script(os.path.join(pipeline_path, "simulate.py"), f"--seed",  f"{seed:d}", "--fgbgR", str(fgbgR), "--snr", str(snrdb))
    # optional arguments: "--eps 1e-5"
    run_script(os.path.join(pipeline_path, "reconstruct.py"),
               "--seed", f"{seed:d}",
               "--method", "composite",
               f"--l2", *(str(u) for u in l2s),
               f"--l1f", *(str(u) for u in l1fs))
    run_script(os.path.join(pipeline_path, "reconstruct.py"),
                "--seed", f"{seed:d}",
                "--method", "blasso",
                f"--lf", *(str(u) for u in lfs))
    # Post process
    run_script(os.path.join(pipeline_path, "postprocess.py"), f"--seed", f"{seed:d}",
               "--fgbgR", str(fgbgR),
               "--snr", str(snrdb),
               f"--l2", *(str(u) for u in l2s),
               f"--l1f", *(str(u) for u in l1fs),
               f"--lf", *(str(u) for u in lfs),
               "--composite",
               "--blasso",
               "--merge",
               "--save")

    # # generate command
    # " ".join([f"--seed", f"{seed:d}",
    #                "--fgbgR", str(fgbgR),
    #                "--snr", str(snrdb),
    #                f"--l2", *(str(u) for u in l2s),
    #                f"--l1f", *(str(u) for u in l1fs),
    #                f"--lf", *(str(u) for u in lfs),
    #                "--composite",
    #                "--blasso",
    #                "--merge",
    #                "--save"])

    #todo save the config parameters in a file and load them directly within the scripts