# 主控脚本：协调训练、测试与可视化

import os
import numpy as np
import torch

from dataloader.generate_data import generate_task_batch
from models.modified_maml import ModifiedMAML, loss_function_maml
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
maml = ModifiedMAML(
    num_ref=NUM_REFS,
    num_sec=NUM_REFS,
    len_c=FILTER_LEN,
    lr=INNER_STEP_SIZE,
    gamma=FORGET_FACTOR,
)
optimizer = torch.optim.SGD(maml.parameters(), lr=META_STEP_SIZE)

for task_idx in range(NUM_TASKS):
    Ref, Di = generate_task_batch(length=LEN_SIGNAL, num_refs=NUM_REFS)
    Fx = torch.from_numpy(Ref[:FILTER_LEN].T).float()
    Fx = Fx.view(NUM_REFS, 1, NUM_ERRS, FILTER_LEN).repeat(1, NUM_REFS, NUM_ERRS, 1)
    Dis = torch.from_numpy(Di[:FILTER_LEN].T).float()

    optimizer.zero_grad()
    anti_noise, gam_vec = maml(Fx, Dis)
    loss = loss_function_maml(anti_noise, Dis, gam_vec)
    loss.backward()
    optimizer.step()
    print(f"[Meta Train] Task {task_idx + 1}/{NUM_TASKS} loss: {loss.item():.4f}")

torch.save(maml.state_dict(), os.path.join(SAVE_DIR, "meta_init.pt"))
print("[Meta Train] Saved meta-initialization.")

# ------------------- Meta-Test (Optional) -------------------
Ref_test, Di_test = generate_task_batch(length=LEN_SIGNAL, num_refs=NUM_REFS)
Fx_test = torch.from_numpy(Ref_test[:FILTER_LEN].T).float()
Fx_test = Fx_test.view(NUM_REFS, 1, NUM_ERRS, FILTER_LEN).repeat(1, NUM_REFS, NUM_ERRS, 1)
Dis_test = torch.from_numpy(Di_test[:FILTER_LEN].T).float()
maml_test = ModifiedMAML(
    num_ref=NUM_REFS,
    num_sec=NUM_REFS,
    len_c=FILTER_LEN,
    lr=INNER_STEP_SIZE,
    gamma=FORGET_FACTOR,
)
maml_test.load_state_dict(maml.state_dict())
with torch.no_grad():
    anti_noise_test, gam_vec_test = maml_test(Fx_test, Dis_test)
    test_loss = loss_function_maml(anti_noise_test, Dis_test, gam_vec_test)
print("[Meta Test] Loss:", float(test_loss))

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
