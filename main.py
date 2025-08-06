# 主控脚本：协调训练、测试与可视化

import numpy as np
import os
from dataloader.generate_data import generate_task_batch
from models.maml_filter import MAMLFilter
from algorithms.fxlms import multi_ref_multi_chan_fxlms
from utils.mat_io import save_mat
from evaluation.mse_plot import compute_mse

# ------------------- Configuration -------------------
FILTER_LEN = 512           # Control filter length
LEN_SIGNAL = FILTER_LEN    # Signal length matches filter length
NUM_TASKS = 10             # Number of meta-training tasks
NUM_REFS = 2               # Reference channels
NUM_ERRS = 1               # Error microphones
META_STEP_SIZE = 1e-2      # Meta learning rate
INNER_STEP_SIZE = 1e-1     # Adaptation step size
FORGET_FACTOR = 0.99       # Forgetting factor
EPSILON = 1.0              # Small weight on gradient

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------- Meta-Training -------------------
maml = MAMLFilter(filter_len=FILTER_LEN, num_refs=NUM_REFS)

for task_idx in range(NUM_TASKS):
    Fx, Di = generate_task_batch(length=LEN_SIGNAL, num_refs=NUM_REFS)
    maml.maml_initial(Fx, Di, mu=INNER_STEP_SIZE, lamda=FORGET_FACTOR, epsilon=EPSILON)
    print(f"[Meta Train] Task {task_idx + 1}/{NUM_TASKS} finished")

# Save meta-trained initialization
np.save(os.path.join(SAVE_DIR, "meta_init.npy"), maml.Phi)
print("[Meta Train] Saved meta-initialization.")

# ------------------- Meta-Test Comparison -------------------
Ref_test, E_test, sec_path = generate_task_batch(length=LEN_SIGNAL,
                                                 num_refs=NUM_REFS,
                                                 with_secondary=True)

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

print("[FxLMS] Final MSE (dB):", mse_base[-1])
print("[MAML+FxLMS] Final MSE (dB):", mse_maml[-1])

# Optionally: save results
save_mat("figures/fxlms_result.mat",
         {"err_base": err_base, "err_maml": err_maml,
          "W_base": W_base, "W_maml": W_maml})
