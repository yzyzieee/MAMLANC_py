# 主控脚本：协调训练、测试与可视化

import numpy as np
import os
from dataloader.generate_data import generate_task_batch
from models.maml_filter import MAMLFilter
from algorithms.MultChanFxLMS import MultChanFxLMS
from utils.mat_io import save_mat
from utils.signal_utils import compute_mse

# ------------------- Configuration -------------------
LEN_SIGNAL = 16000         # Signal length (1 sec at 16 kHz)
FILTER_LEN = 512           # Control filter length
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
    maml.adapt(Fx, Di, mu=INNER_STEP_SIZE, lamda=FORGET_FACTOR, epsilon=EPSILON)
    print(f"[Meta Train] Task {task_idx + 1}/{NUM_TASKS} finished")

# Save meta-trained initialization
np.save(os.path.join(SAVE_DIR, "meta_init.npy"), maml.Phi)
print("[Meta Train] Saved meta-initialization.")

# ------------------- Meta-Test (Optional) -------------------
# Adapt to a new task
Fx_test, Di_test = generate_task_batch(length=LEN_SIGNAL, num_refs=NUM_REFS)
maml_test = MAMLFilter(filter_len=FILTER_LEN, num_refs=NUM_REFS)
maml_test.Phi = maml.Phi.copy()
maml_test.adapt(Fx_test, Di_test, mu=INNER_STEP_SIZE, lamda=FORGET_FACTOR, epsilon=EPSILON)

# FxLMS baseline using the class-based implementation
Ref_test, E_test, sec_path = generate_task_batch(
    length=LEN_SIGNAL, num_refs=NUM_REFS, with_secondary=True
)
fxlms = MultChanFxLMS(
    ref_num=NUM_REFS,
    err_num=NUM_ERRS,
    ctrl_num=NUM_REFS,
    filter_len=FILTER_LEN,
    sec_path=sec_path,
    stepsize=INNER_STEP_SIZE,
)
_, err_fxlms = fxlms.process_batch(Ref_test, E_test)
W_fxlms = fxlms.weights

# MSE evaluation
mse_fxlms = compute_mse(err_fxlms)
print("[FxLMS] MSE (dB):", mse_fxlms)

# Optionally: save results
save_mat("figures/fxlms_result.mat", {"err": err_fxlms, "W": W_fxlms})
