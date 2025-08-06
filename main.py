# 主控脚本：协调训练、测试与可视化

import os
import numpy as np

from dataloader.generate_data import (
    generate_anc_training_data,
    generate_task_batch,
)
from models.maml_filter import MAMLFilter
from algorithms.fxlms import multi_ref_multi_chan_fxlms
from utils.mat_io import save_mat
from evaluation.mse_plot import compute_mse, plot_mse_compare

# ------------------- Configuration -------------------
FILTER_LEN = 512           # Control filter length
LEN_SIGNAL = FILTER_LEN    # Signal length matches filter length
NUM_TASKS = 10             # Number of meta-training epochs
NUM_REFS = 1               # Single reference channel
NUM_ERRS = 1               # Error microphones
META_STEP_SIZE = 1e-2      # Meta learning rate
INNER_STEP_SIZE = 1e-1     # Adaptation step size
FORGET_FACTOR = 0.99       # Forgetting factor
EPSILON = 1.0              # Small weight on gradient

# Dataset paths (modify as needed)
PATH_DIR = "E:/NTU/Test/HeadphonePathMeasurement/Recording/Primary/PriPathModels_E"
TRAIN_FILES = [
    "PriPath_Pri_d50_0.mat", "PriPath_Pri_d100_90.mat",
    "PriPath_Pri_d100_135.mat", "PriPath_Pri_d50_225.mat",
    "PriPath_Pri_d100_270.mat", "PriPath_Pri_d20_45.mat",
]
SEC_PATH_FILE = (
    "E:/NTU/Test/HeadphonePathMeasurement/Recording/Secondary/"
    "SecPathModels/SecPath_firmed.mat"
)

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------- Meta-Training -------------------
maml = MAMLFilter(filter_len=FILTER_LEN, num_refs=NUM_REFS)

# Generate training samples from measured paths
Fx_data, Di_data = generate_anc_training_data(
    PATH_DIR, TRAIN_FILES, SEC_PATH_FILE, NUM_TASKS, FILTER_LEN
)

for task_idx in range(NUM_TASKS):
    Fx = Fx_data[:, task_idx].reshape(FILTER_LEN, NUM_REFS)
    Di = Di_data[:, task_idx].reshape(FILTER_LEN, 1)
    maml.maml_initial(Fx, Di, mu=INNER_STEP_SIZE, lamda=FORGET_FACTOR, epsilon=EPSILON)
    print(f"[Meta Train] Task {task_idx + 1}/{NUM_TASKS} finished")

# Save meta-trained initialization
np.save(os.path.join(SAVE_DIR, "meta_init.npy"), maml.Phi)
print("[Meta Train] Saved meta-initialization.")

# ------------------- Meta-Test Comparison -------------------
Ref_test, E_test, sec_path = generate_task_batch(
    length=LEN_SIGNAL, num_refs=NUM_REFS, with_secondary=True
)

# Number of control loudspeakers inferred from secondary path
CTRL_NUM = sec_path.shape[1] // NUM_ERRS

# Baseline FxLMS (zero initialisation)
W_base, err_base = multi_ref_multi_chan_fxlms(Ref_test, E_test,
                                              FILTER_LEN, sec_path,
                                              INNER_STEP_SIZE)

# MAML-initialised FxLMS
W_single = maml.Phi.reshape(FILTER_LEN, NUM_REFS)
# replicate initial weights for each control channel
W_init = np.tile(W_single, (1, CTRL_NUM))
W_maml, err_maml = multi_ref_multi_chan_fxlms(Ref_test, E_test,
                                              FILTER_LEN, sec_path,
                                              INNER_STEP_SIZE,
                                              init_W=W_init)

# MSE evaluation
mse_base = compute_mse(err_base.flatten())
mse_maml = compute_mse(err_maml.flatten())

plot_mse_compare(
    {"FxLMS": mse_base, "MAML+FxLMS": mse_maml},
    "MSE Convergence",
    save_path="figures/mse_compare.png",
)

print("[FxLMS] Final MSE (dB):", mse_base[-1])
print("[MAML+FxLMS] Final MSE (dB):", mse_maml[-1])

# Optionally: save results
save_mat("figures/fxlms_result.mat",
         {"err_base": err_base, "err_maml": err_maml,
          "W_base": W_base, "W_maml": W_maml})
