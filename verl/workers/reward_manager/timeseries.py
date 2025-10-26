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
from typing import List, Tuple, Optional
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Global counter for saving generated code
_code_counter = 0

def clean_generated_code(code: str) -> str:
    """Clean generated code by removing markdown and chat tokens"""
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    code = re.sub(r'<\|im_end\|>', '', code)
    code = re.sub(r'<\|im_start\|>', '', code)
    code = re.sub(r'<\|.*?\|>', '', code)
    return code.strip()

def execute_code_safely(code: str, historical_data: List[float], 
                        future_count: int, timeout: int = 30) -> Tuple[float, Optional[np.ndarray], str]:
    """Execute generated code safely and return predictions"""
    global _code_counter
    _code_counter += 1

    code = clean_generated_code(code)
    
    os.makedirs("generated_codes", exist_ok=True)
    with open(f"generated_codes/code_{_code_counter:04d}.py", 'w') as f:
        f.write(code)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = os.path.join(temp_dir, "forecast_code.py")
            with open(code_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                [sys.executable, code_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            submission_path = os.path.join(temp_dir, "submission.csv")

            if os.path.exists(submission_path):
                os.makedirs("generated_submission", exist_ok=True)
                shutil.copy2(submission_path, f"generated_submission/submission_{_code_counter:04d}.csv")

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
                    return 0.1, None, f"Failed to read predictions: {str(e)}"
            else:
                return 0.1, None, f"No submission file created. stderr: {result.stderr[:200]}"
                
    except subprocess.TimeoutExpired:
        return 0, None, "Code execution timed out"
    except Exception as e:
        return 0, None, f"Execution error: {str(e)}"

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
                    response_str, historical_data, future_count
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