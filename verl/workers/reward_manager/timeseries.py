# Copyright 2025 Zhejiang University (ZJU), China and TimeMaster Team.

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
from datetime import datetime
from typing import List, Tuple, Optional
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Global counter for saving generated code
_code_counter = 0

root_folder = f'/fsx/s3/chronos-o1/verl/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

def _log_error(code_id: int, error_msg: str, stderr: str):
    """Log execution errors to file"""
    log_dir = os.path.join(root_folder, "execution_logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"error_{code_id:04d}.txt"), 'w') as f:
        f.write(f"Error: {error_msg}\n")
        f.write(f"Stderr: {stderr}\n")

def clean_generated_code(code: str) -> str:
    """Extract Python script from LLM output"""
    python_blocks = re.findall(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
    if python_blocks:
        code = python_blocks[0]
    
    return code.strip()

def execute_code_safely(code: str, historical_data: List[float], 
                        future_count: int, train_dir: str = "", val_dir: str = "", timeout: int = 30) -> Tuple[float, Optional[np.ndarray], str]:
    """Execute generated code safely and return predictions"""
    global _code_counter
    _code_counter += 1

    code = clean_generated_code(code)
    
    codes_dir = os.path.join(root_folder, "generated_codes")
    os.makedirs(codes_dir, exist_ok=True)
    with open(os.path.join(codes_dir, f"code_{_code_counter:04d}.py"), 'w') as f:
        f.write(code)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = os.path.join(temp_dir, "forecast_code.py")
            with open(code_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                [sys.executable, code_file, "--train_dir", train_dir, "--val_dir", val_dir],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            submission_path = os.path.join(temp_dir, "submission.csv")

            if os.path.exists(submission_path):
                submission_dir = os.path.join(root_folder, "generated_submission")
                os.makedirs(submission_dir, exist_ok=True)
                shutil.copy2(submission_path, os.path.join(submission_dir, f"submission_{_code_counter:04d}.csv"))

                try:
                    pred_df = pd.read_csv(submission_path)
                    predictions = pred_df.iloc[:, -1].values
                    
                    if len(predictions) != future_count:
                        if len(predictions) > future_count:
                            predictions = predictions[:future_count]
                        else:
                            pad_value = predictions[-1] if len(predictions) > 0 else np.mean(historical_data) if historical_data else 0.0
                            predictions = np.concatenate([
                                predictions, 
                                [pad_value] * (future_count - len(predictions))
                            ])
                        return 0.5, predictions, "Prediction length mismatch, adjusted"

                    elif np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                        predictions[np.isnan(predictions)] = 0.0
                        predictions[np.isinf(predictions)] = 0.0
                        return 0.8, predictions, "Predictions contain NaN or Inf"
                    
                    else:
                        return 1.0, predictions, "Success"
                    
                except Exception as e:
                    error_msg = f"Failed to read predictions: {str(e)}"
                    _log_error(_code_counter, error_msg, result.stderr)
                    return 0.1, None, error_msg
            else:
                error_msg = f"No submission file created. stderr: {result.stderr[:200]}"
                _log_error(_code_counter, error_msg, result.stderr)
                return 0.1, None, error_msg
                
    except subprocess.TimeoutExpired:
        error_msg = "Code execution timed out"
        _log_error(_code_counter, error_msg, "")
        return 0, None, error_msg
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        _log_error(_code_counter, error_msg, "")
        return 0, None, error_msg

def calculate_forecasting_reward(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate reward based on forecasting accuracy"""
    assert len(predictions) == len(ground_truth), "Prediction and ground truth lengths do not match"
    
    mse = np.mean((predictions - ground_truth) ** 2)
    scale_factor = 10000  
    accuracy_reward = max(0, 1 - mse / scale_factor)
    
    return accuracy_reward

@register("timeseries")
class TimeSeriesRewardManager(AbstractRewardManager):
    """Reward manager for time series code generation tasks"""
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.buffer = np.zeros(100, dtype=bool)
        self.save_idx = 0
        
    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        execution_successes = []
        already_printed = 0

        for i in range(len(data)):
            data_item = data[i]
            
            # Decode prompt and response
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get data from extra_info in parquet
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            historical_data = extra_info.get('past_data', [])
            ground_truth = extra_info.get('future_data', [])
            train_dir = data_item.non_tensor_batch.get('train_dir', '')
            val_dir = data_item.non_tensor_batch.get('val_dir', '')
            
            if isinstance(historical_data, str):
                historical_data = ast.literal_eval(historical_data)
            if isinstance(ground_truth, str):
                ground_truth = ast.literal_eval(ground_truth)
            if isinstance(ground_truth, list):
                ground_truth = np.array(ground_truth)
            
            future_count = len(ground_truth)
            
            # Execute code
            try:
                exe_reward, predictions, error_msg = execute_code_safely(
                    response_str, historical_data, future_count, train_dir, val_dir
                )
                
                execution_success = exe_reward > 0.5
                execution_successes.append(execution_success)
                
                self.buffer[self.save_idx % len(self.buffer)] = execution_success
                self.save_idx += 1
                
                # Calculate accuracy reward
                if predictions is None:
                    accuracy_reward = 0.0
                    predictions = np.zeros(future_count)
                else:
                    try:
                        accuracy_reward = calculate_forecasting_reward(predictions, ground_truth)
                    except Exception as e:
                        accuracy_reward = 0.0
                        predictions = np.zeros(future_count)
                
                # Weighted combination: 50% execution, 50% accuracy
                total_score = 0.5 * exe_reward + 0.5 * accuracy_reward
                
                reward_tensor[i, valid_response_length - 1] = total_score
                
                # Store extra info
                reward_extra_info["execution_success"].append(execution_success)
                reward_extra_info["accuracy_reward"].append(accuracy_reward)
                reward_extra_info["total_score"].append(total_score)
                
            except Exception as e:
                reward_tensor[i, valid_response_length - 1] = 0.0
                execution_successes.append(False)
                reward_extra_info["execution_success"].append(False)
                reward_extra_info["accuracy_reward"].append(0.0)
                reward_extra_info["total_score"].append(0.0)
            
            # Print examples for debugging
            if already_printed < self.num_examine:
                already_printed += 1
                print("[Time Series Code Generation Reward]")
                print("[prompt]", prompt_str[:200] + "..." if len(prompt_str) > 200 else prompt_str)
                print("[response]", response_str[:200] + "..." if len(response_str) > 200 else response_str)
                print("[execution_success]", execution_successes[-1])
                print("[total_score]", reward_tensor[i, valid_response_length - 1].item())
        
        # Print format success rate
        if self.save_idx >= 100:
            print("[Format success rate]:", self.buffer.mean())
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor