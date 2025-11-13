import os
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
import torch
import re
import shutil
import ast
import uuid
from typing import List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# --- safer default for output dir ---
_root_env = os.environ.get('VERL_OUTPUT_DIR')
root_folder = (_root_env if _root_env else '/tmp/verl_outputs') + '/'
os.makedirs(root_folder, exist_ok=True)

MAX_CONCURRENT = 16  # cap parallel executions

# ---------- logging / helpers ----------
def _log_error(job_id: str, error_msg: str, stderr: str):
    os.makedirs(root_folder + "execution_logs", exist_ok=True)
    with open(root_folder + f"execution_logs/error_{job_id}.txt", 'w') as f:
        f.write(f"Error: {error_msg}\n")
        if stderr:
            f.write(f"Stderr: {stderr}\n")

def clean_generated_code(code: str) -> str:
    block = re.findall(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
    return (block[0] if block else code).strip()

# ---------- EXECUTOR (now receives job_id) ----------
def execute_code_safely(code: str,
                        job_id: str,
                        historical_data: List[float],
                        future_count: int,
                        train_dir: str = "",
                        val_dir: str = "",
                        timeout: int = 600) -> Tuple[float, Optional[np.ndarray], str]:
    """
    Run code that must write submission.csv; copy artifacts using job_id.
    Returns (exe_reward, predictions or None, message).
    """
    code = clean_generated_code(code)

    # persist code for this job
    os.makedirs(root_folder + "generated_codes", exist_ok=True)
    with open(root_folder + f"generated_codes/code_{job_id}.py", 'w') as f:
        f.write(code)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = os.path.join(temp_dir, "forecast_code.py")
            with open(code_file, 'w') as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, code_file, "--train_dir", train_dir, "--val_dir", val_dir],
                cwd=temp_dir, capture_output=True, text=True, timeout=timeout
            )

            submission_path = os.path.join(temp_dir, "submission.csv")
            if not os.path.exists(submission_path):
                err = f"No submission.csv created. stderr: {result.stderr[:200]}"
                _log_error(job_id, err, result.stderr)
                return 0.1, None, err

            os.makedirs(root_folder + "generated_submission", exist_ok=True)
            shutil.copy2(submission_path, root_folder + f"generated_submission/submission_{job_id}.csv")

            try:
                pred_df = pd.read_csv(submission_path)
                predictions = pred_df.iloc[:, -1].values

                if len(predictions) != future_count:
                    if len(predictions) > future_count:
                        predictions = predictions[:future_count]
                    else:
                        pad_value = predictions[-1] if len(predictions) > 0 else (
                            np.mean(historical_data) if len(historical_data) > 0 else 0.0
                        )
                        predictions = np.concatenate([predictions, [pad_value] * (future_count - len(predictions))])
                    return 0.5, predictions, "Prediction length mismatch, adjusted"

                if np.any(~np.isfinite(predictions)):
                    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
                    return 0.8, predictions, "Predictions contained NaN/Inf"

                return 1.0, predictions, "Success"

            except Exception as e:
                err = f"Failed to read predictions: {e}"
                _log_error(job_id, err, result.stderr)
                return 0.1, None, err

    except subprocess.TimeoutExpired:
        err = "Code execution timed out"
        _log_error(job_id, err, "")
        return 0.0, None, err
    except Exception as e:
        err = f"Execution error: {e}"
        _log_error(job_id, err, "")
        return 0.0, None, err

# ---------- reward (unchanged from your latest version) ----------
def calculate_forecasting_reward(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    historical_data: np.ndarray,
    *,
    tail_gamma: float = 1.25,
    min_var_frac: float = 0.05,
    use_delta: bool = True,
    delta_weight: float = 0.10,
    var_weight: float = 0.30,
) -> float:
    assert len(predictions) == len(ground_truth), "Prediction and ground truth lengths do not match"

    T = len(ground_truth)
    idx = np.arange(1, T + 1, dtype=float)
    w = idx ** float(tail_gamma)
    w = w / (w.sum() + 1e-12)

    hist = np.asarray(historical_data, dtype=float)
    mad_lvl = np.mean(np.abs(hist - hist.mean())) + 1e-8

    mae_lvl = np.sum(w * np.abs(predictions - ground_truth))
    base_acc_lvl = np.exp(-mae_lvl / mad_lvl)

    baseline_lvl = np.full_like(ground_truth, hist[-1])
    baseline_mae_lvl = np.sum(w * np.abs(baseline_lvl - ground_truth))
    baseline_acc_lvl = np.exp(-baseline_mae_lvl / mad_lvl)

    advantage_lvl = base_acc_lvl - baseline_acc_lvl
    adv_term = 1.0 / (1.0 + np.exp(-5.0 * advantage_lvl))

    delta_adv_term = 0.0
    if use_delta and T > 0:
        start_val = hist[-1]
        pred_d = np.diff(np.concatenate([[start_val], predictions]))
        gt_d   = np.diff(np.concatenate([[start_val], ground_truth]))
        hist_d = np.diff(hist) if len(hist) > 1 else np.array([0.0], dtype=float)
        mad_d = (np.mean(np.abs(hist_d - hist_d.mean())) if len(hist_d) > 0 else 1.0) + 1e-8

        mae_d = np.sum(w * np.abs(pred_d - gt_d))
        base_acc_d = np.exp(-mae_d / mad_d)

        baseline_d = np.zeros_like(gt_d)
        baseline_mae_d = np.sum(w * np.abs(baseline_d - gt_d))
        baseline_acc_d = np.exp(-baseline_mae_d / mad_d)

        advantage_d = base_acc_d - baseline_acc_d
        delta_adv_term = 1.0 / (1.0 + np.exp(-5.0 * advantage_d))

    hist_var = np.var(hist) + 1e-8
    pred_var = np.var(predictions)
    required = min_var_frac * hist_var
    var_penalty = max(0.0, (required - pred_var) / required)
    var_term = np.exp(-var_penalty)

    delta_w = (delta_weight if use_delta else 0.0)
    var_w = max(0.0, min(1.0, var_weight))
    adv_w = max(0.0, 1.0 - var_w - delta_w)

    reward = adv_w * adv_term + delta_w * delta_adv_term + var_w * var_term
    return float(np.clip(reward, 0.0, 1.0))

@register("timeseries")
class TimeSeriesRewardManager(AbstractRewardManager):
    """Reward manager for time series code generation tasks (parallel, per-job artifacts)."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.buffer = np.zeros(100, dtype=bool)
        self.save_idx = 0

    def __call__(self, data: DataProto, return_dict: bool = False):
        device = data.batch['responses'].device
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32, device=device)
        reward_extra_info = defaultdict(list)

        # -------- 1) Build job list with UNIQUE job_id per item --------
        jobs = []
        for i in range(len(data)):
            item = data[i]

            prompt_ids = item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            attn = item.batch['attention_mask']
            if torch.is_tensor(attn):
                valid_prompt_length  = int(attn[:prompt_length].sum().item())
                valid_response_length = int(attn[prompt_length:].sum().item())
            else:
                valid_prompt_length  = int(attn[:prompt_length].sum())
                valid_response_length = int(attn[prompt_length:].sum())

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = item.batch['responses']
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            extra_info = item.non_tensor_batch.get('extra_info', {})
            historical_data = extra_info.get('past_data', [])
            ground_truth = extra_info.get('future_data', [])
            train_dir = item.non_tensor_batch.get('train_dir', '')
            val_dir = item.non_tensor_batch.get('val_dir', '')

            if isinstance(historical_data, str):
                historical_data = ast.literal_eval(historical_data)
            if isinstance(ground_truth, str):
                ground_truth = ast.literal_eval(ground_truth)
            if isinstance(ground_truth, list):
                ground_truth = np.array(ground_truth, dtype=float)

            job_id = f"{uuid.uuid4().hex[:8]}_{i}"  # UNIQUE per item, thread-safe

            jobs.append({
                "idx": i,
                "job_id": job_id,
                "valid_response_length": valid_response_length,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "historical_data": np.array(historical_data, dtype=float),
                "ground_truth": ground_truth,
                "future_count": len(ground_truth),
                "train_dir": train_dir,
                "val_dir": val_dir,
            })

        # -------- 2) Run in waves with max concurrency --------
        prints = 0
        for start in range(0, len(jobs), MAX_CONCURRENT):
            batch = jobs[start:start + MAX_CONCURRENT]
            with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT, len(batch))) as ex:
                futures = {ex.submit(
                    execute_code_safely,
                    j["response_str"], j["job_id"], j["historical_data"], j["future_count"], j["train_dir"], j["val_dir"]
                ): j for j in batch}

                for fut, j in futures.items():
                    try:
                        exe_reward, predictions, error_msg = fut.result()
                    except Exception as e:
                        exe_reward, predictions, error_msg = 0.0, None, f"Executor exception: {e}"
                        _log_error(j["job_id"], error_msg, "")

                    # --- execution gate: small negative on failure; else accuracy-only ---
                    if predictions is None:
                        accuracy_reward = 0.0
                        total_score = -0.05
                        execution_success = False
                    else:
                        try:
                            accuracy_reward = calculate_forecasting_reward(
                                predictions, j["ground_truth"], j["historical_data"],
                                tail_gamma=1.25, min_var_frac=0.05,
                                use_delta=True, delta_weight=0.10, var_weight=0.30
                            )
                        except Exception as e:
                            accuracy_reward = 0.0
                        total_score = float(accuracy_reward)
                        execution_success = True

                    # place reward on last token
                    vr = j["valid_response_length"]
                    if vr > 0:
                        reward_tensor[j["idx"], vr - 1] = total_score

                    # telemetry
                    self.buffer[self.save_idx % len(self.buffer)] = execution_success
                    self.save_idx += 1

                    reward_extra_info["job_id"].append(j["job_id"])
                    reward_extra_info["execution_success"].append(execution_success)
                    reward_extra_info["accuracy_reward"].append(float(accuracy_reward))
                    reward_extra_info["total_score"].append(float(total_score))

                    # limited prints
                    if prints < self.num_examine:
                        prints += 1
                        print("[Time Series Code Generation Reward]")
                        print("[job_id]", j["job_id"])
                        print("[prompt]", j["prompt_str"][:200] + "..." if len(j["prompt_str"]) > 200 else j["prompt_str"])
                        print("[response]", j["response_str"][:200] + "..." if len(j["response_str"]) > 200 else j["response_str"])
                        print("[execution_success]", execution_success)
                        print("[total_score]", total_score)

        if self.save_idx >= 100:
            print("[Format success rate]:", float(self.buffer.mean()))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
