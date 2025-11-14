"""
CiK code generation dataset creation from raw data
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
import random
import yaml
from verl.ts_models.ts_forecasting_template import TSFM_TEMPLATES, LLM_FORECASTING_TEMPLATES, UNIMODAL_FORECASTING_TEMPLATES

TSFM_POOL = [
    # chronos
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
    "amazon/chronos-t5-large",
    # chronos-bolt
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
    # chronos-2
    "amazon/chronos-2",
]
UNIMODAL_POOL = [
    "PatchTST",
    "DLinear",
    "iTransformer",
]
LLM_POOL = [
    "gpt2",
]
MODEL_POOL = [
    TSFM_POOL,
    UNIMODAL_POOL,
    LLM_POOL
]

def get_code_files(sft_dir):
    """Get all Python files from sft directory"""
    code_files = []
    for root, dirs, files in os.walk(sft_dir):
        for file in files:
            if file.endswith('.py'):
                code_files.append(os.path.join(root, file))
    return code_files

def get_random_code_sample(code_files):
    """Randomly sample code from pre-built code files list"""
    if not code_files:
        return "# No code files found"
    
    selected_file = random.choice(code_files)
    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "# Error reading code file"

def create_code_generation_example(past_data, future_data, context, task_folder, sample_idx, code_files, data_split, base_data_path):
    """Create code generation example from raw data"""
    # Extract timestamps, covariates, and target values
    timestamps = past_data.iloc[:, 0].tolist()
    past_values = [round(x, 2) for x in past_data.iloc[:, -1].values.tolist()]
    future_values = [round(x, 2) for x in future_data.iloc[:, -1].values.tolist()]
    
    # Extract covariates (columns between timestamps and target)
    covariate_names = []
    covariate_values = []
    if past_data.shape[1] > 2:
        covariate_names = past_data.columns[1:-1].tolist()
        for i in range(len(timestamps)):
            cov_values = [round(past_data.iloc[i, j], 2) for j in range(1, past_data.shape[1]-1)]
            covariate_values.append(cov_values)
    
    selected_model_type = random.choice(MODEL_POOL)
    selected_model = random.choice(selected_model_type)
    if selected_model_type == TSFM_POOL:
        selected_template = TSFM_TEMPLATES(selected_model)
    elif selected_model_type == UNIMODAL_POOL:
        selected_template = UNIMODAL_FORECASTING_TEMPLATES(selected_model)
    elif selected_model_type == LLM_POOL:
        selected_template = LLM_FORECASTING_TEMPLATES(selected_model)
    
    # Create input prompt
    input_prompt = f"""Multimodal Time Series Forecasting Challenge

## Task Description
You are tasked with developing a multimodal time series forecasting model that leverages both numerical time series data and textual context to predict future values. This challenge combines temporal patterns with semantic information to improve forecasting accuracy.

## Dataset Description
- You will have access to training data provided as named argument `--train_dir`, which is a directory containing training samples in numbered folders (1/, 2/, 3/, etc.)
- Each training sample folder contains a `past_time.csv` for historical data (1st column=timestamp, last column=target, middle columns=covariates), a `future_time.csv` for future data to predict (same format as past_time.csv), a `context` or `context.csv` for context (`context` for static text context for the entire series, OR `context.csv` for time-aligned dynamic context if context changes over time, check if context or context.csv exists and load correspondingly)
- You will also have access to the validation data provided as named argument `--val_dir`, which is the same format as `train_dir` except removing `future_time.csv` to avoid data leakage, so do not attempt to load `future_time.csv` from `val_dir`
- Preview of `past_time.csv` in one sample folder of `val_dir`: target values {past_values[:100]}, covariate names = {covariate_names if covariate_names else 'None'}, covariate values = {covariate_values if covariate_values else 'None'}
- Preview of context in one sample folder of `val_dir`: {context[:500]}

## Objective
Predict future values for all validation samples using both historical numerical patterns and textual context information. Load `future_time.csv` in `train_dir` to get future length.

## Evaluation Metric
Models will be evaluated using Mean Squared Error (MSE) between predicted and actual future values:
MSE = (1/n) * Σ(y_true - y_pred)²

Lower MSE values indicate better performance.

## Submission Format
Submit predictions in a CSV file named `submission.csv`:
- No timestamps needed, just the predicted values

## Key Considerations:
1. **Data Loading**: Correctly load the validation data from the given `val_dir`. Do not copy paste the above preview time series and context in the generated code.
2. **Context Enhancement**: Consider the context file to enhance forecasting
3. **Forecasting Horizon**: Predict values for the exact number of time steps as in future_time.csv for each sample
4. **Feature Engineering**: Consider effective feature preprocessing to extract temporal patterns and semantic relationships from context

## Modeling Approaches
You can use any combination of the following approaches. You have access to GPU so make sure to load models on GPU when possible for faster computations. Consider different strategies across iterations to increase solution diversity:

### 1. Feature Preprocessing
- Covariate selection
- Trend and seasonality decomposition
- Frequency domain transformation (FFT, wavelets)
- Autocorrelation analysis
- Normalization and scaling
- Outlier detection and handling
- Missing value imputation

### 2. Context Selection
- Select important context
- Context summarization
- If context is dynamically aligned with timestamps, you can select important timestamps to use corresponding context
- You can also choose to not use the context at all if you find it not helpful for forecasting

### 3. Model Architecture Options

#### A. Time Series Foundation Models (TSFMs)
- Chronos, Moirai, TimesFM
- Use provided templates for correct implementation
        
#### B. Train Alignment-based Multimodal Models
- Find effective time series encoders: Transformer (e.g., PatchTST, Chronos, FEDformer, TimesFM, Autoformer), MLP (e.g., DLinear), CNN (e.g., TimesNet), RNN (e.g., SegRNN)
- Find effective text encoders: bert, sentence bert, gpt, llama, qwen, mistral, t5, flan-t5, bart, palm, bge
- Fusion strategies: early/middle/late fusion
- Fusion operations: concatenation, addition, attention mechanisms
- Training strategies: frozen, efficient fine-tuning, full fine-tuning
- Custom loss functions and hyperparameter optimization

#### C. Prompting-based Approaches
- Claude 3.7 (use us.anthropic.claude-3-7-sonnet-20250219-v1:0)
- Open-source LLMs on HuggingFace such as gpt, llama, qwen, mistral
- Custom prompting strategies

#### D. Train Unimodal Time Series Models
- PatchTST, DLinear, FEDformer, Autoformer, Crossformer, FiLM, Informer, iTransformer, FreTS, LightTS, Nonstationary_Transformer, SegRNN, TiDE, TimesNet, Transformer, TSMixer

### 4. Model Ensemble
- Decide whether to ensemble multiple models
- Weighted averaging, stacking, or voting strategies

### 5. Forecast Refinement
- Post-processing based on textual context
- Context-aware forecast correction

## Data Preprocessing:
- Load training samples from `train_dir` if model training/fine-tuning, note that you may or may not use `train_dir` depending on if training is involved. If training, you should train until convergence with a time limit of 10 minutes.
- Read and parse the validation time series and context text from `val_dir`. Do not copy paste the above preview time series and context in the generated code.
- Appropriate feature preprocessing
- Process text data (context selection or summarization if needed, tokenization, encoding)
- Ensure predictions match the length of `future_time.csv` in `train_dir`

## Implementation Requirements:
- Take two named arguments: `--train_dir` and `--val_dir`
- Train on all samples in `train_dir` (You may or may not use it depending on if training is needed. If training, you should train until convergence with a time limit of 10 minutes.)
- Generate predictions for ALL validation samples in `val_dir`
- Concatenate predictions from all validation samples in order (sample 1, 2, 3, etc.)
- Save concatenated results as `submission.csv` in the current directory

## Additional Notes:
- Example code to load data. Make sure to load target time series as floating-point type.
```python
past_data = pd.read_csv(os.path.join(sample_path, 'past_time.csv')).values
past_target = past_data[:, -1].astype(np.float32)
```
- The prediction length must match the length of `future_time.csv` in `train_dir`. The `submission.csv` must have the same format as `future_time.csv` where each row corresponds to a time step.
- Some text embedding models have maximum token length so you need to select the most important part of context as input to the text embedding model. For example, maximum token length for `bert-base-uncased` is 512.
- Time series forecasting means predicting all future values across the forecasting horizon, not just a single next step. For regression-based approaches, use models like MultiOutputRegressor to predict multiple steps simultaneously. The model should take the entire past time series as input and output the full sequence of future values in one forward pass.

## Example Model:
Below is an example model code snippet for reference: {selected_template}

Remember to avoid future information leakage in your model development process. Do not overthink and prioritize generating Python script."""
    
    # Get random code sample
    random_code = get_random_code_sample(code_files)
    
    # Create example in required format
    example = {
        "extra_info": {
            "question": input_prompt,
            "answer": random_code,
            'past_data': past_values,
            'future_data': future_values,
            'timestamps': timestamps,
            'covariate_names': covariate_names,
            'covariate_values': covariate_values,
            'context': context,
        },
        "prompt": [
            {
                "role": "user",
                "content": input_prompt
            }
        ],
        "train_dir": f"{base_data_path}/train",
        "val_dir": f"{base_data_path}/{data_split}/{sample_idx}"
    }

    return example

def build_code_dataset(data_dir, code_files, data_split, base_data_path):
    """Build code generation dataset for train/test split"""
    examples = []
    
    for sample_idx in sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()], key=int):
        sample_path = Path(data_dir) / str(sample_idx)
        
        if not sample_path.exists():
            continue
            
        # Read data files
        past_data = pd.read_csv(sample_path / 'past_time.csv')
        future_data = pd.read_csv(sample_path / 'future_time.csv')
        
        with open(sample_path / 'context', 'r') as f:
            context = f.read().strip()
        
        # Create code generation example
        example = create_code_generation_example(
            past_data, future_data, context, str(data_dir), sample_idx, code_files, data_split, train_dir_path
        )
        examples.append(example)

    return examples

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Create output directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Build code files list once
    code_files = get_code_files(config['sft_dir'])
    
    # Get all task folders
    data_path = Path(config['cik_dir'])
    train_dir = data_path / 'val'
    test_dir = data_path / 'test'
    
    # Build datasets
    train_examples = build_code_dataset(train_dir, code_files, 'val', config['base_data_path'])
    test_examples = build_code_dataset(test_dir, code_files, 'test', config['base_data_path'])
    
    # Save as parquet files
    train_df = pd.DataFrame(train_examples)
    test_df = pd.DataFrame(test_examples)
    
    train_df.to_parquet(output_dir / config['train_parquet_name'], index=False)
    test_df.to_parquet(output_dir / config['test_parquet_name'], index=False)
    
    print(f"Parquet files saved to {output_dir}")
    print(f"Train samples: {len(train_examples)}")
    print(f"Test samples: {len(test_examples)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_config = Path(__file__).parent / 'config.yaml'
    parser.add_argument('--config', type=str, default=str(default_config), help='Path to YAML configuration file')
    args = parser.parse_args()
    main(args)