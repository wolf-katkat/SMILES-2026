import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)

tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx, relaxation_param=1.8):
    """
    SART (Simultaneous Algebraic Reconstruction Technique).
    """
    score_filter = helpers["score_filter"]
    fit_tx_prediction = helpers["fit_tx_prediction"]

    tx_part = fit_tx_prediction(rx)
    residual = rx - tx_part

    res_band = np.column_stack([
        score_filter(residual[:, ch]) for ch in range(residual.shape[1])
    ])

    # Extract Spatial Rank-1 component
    cov = res_band.conj().T @ res_band / res_band.shape[0]
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    shared = res_band @ v
    denom = np.vdot(shared, shared) + 1e-30
    rank1_part = np.column_stack([
        (np.vdot(shared, res_band[:, ch]) / denom) * shared
        for ch in range(res_band.shape[1])
    ])

    target = tx_part + rank1_part

    delta = np.zeros(rx.shape[0], dtype=np.complex128)
    center_idx = rx.shape[0] // 2
    delta[center_idx] = 1.0

    #Get a filter
    h_kernel = score_filter(delta)

    #Calculate the sum in raws and columns (SART-normalization)
    l1_norm = np.sum(np.abs(h_kernel))

    sart_alpha = relaxation_param / (l1_norm ** 2)

    def apply_H(x):
        return np.column_stack([
            score_filter(x[:, ch]) for ch in range(x.shape[1])
        ])

    def apply_H_adjoint(x):
        x_rev_conj = np.flip(x, axis=0).conj()
        filtered = apply_H(x_rev_conj)
        return np.flip(filtered, axis=0).conj()

    # SART 
    iterations = 2  
    x_hat = target.copy()

    for _ in range(iterations):
        # 1. Calculate the predicted error
        err = target - apply_H(x_hat)

        # Refresh the vector usinf SART estimation of the error
        x_hat = x_hat + sart_alpha * apply_H_adjoint(err)

    return rx - x_hat

print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
