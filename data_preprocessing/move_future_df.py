#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def move_future_df_files():
    """
    For each sample folder under /fsx/mmts/raw/val/i, move its future_df.csv 
    to the corresponding folder under /fsx/mmts/raw/val_gt/i
    """
    val_dir = Path("/fsx/mmts/raw/test")
    val_gt_dir = Path("/fsx/mmts/raw/test_gt")
    
    if not val_dir.exists():
        print(f"Source directory {val_dir} does not exist")
        return
    
    # Create val_gt directory if it doesn't exist
    val_gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all sample directories (numeric folders)
    sample_dirs = [d for d in val_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        future_df_path = sample_dir / "0" / "future_time.csv"
        
        # Skip if future_df.csv doesn't exist
        if not future_df_path.exists():
            print(f"future_time.csv not found in {sample_dir}")
            continue
        
        # Create corresponding directory in val_gt
        dest_dir = val_gt_dir / sample_name / "0"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Move future_df.csv
        dest_path = dest_dir / "future_time.csv"
        shutil.move(str(future_df_path), str(dest_path))
        
        print(f"Moved {future_df_path} to {dest_path}")
    
    print("File moving completed")

if __name__ == "__main__":
    move_future_df_files()