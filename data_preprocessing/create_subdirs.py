#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def create_subdirectories():
    """
    For each sample folder under /fsx/mmts/raw/test/, create a subdirectory 
    named "0" and move existing files into it.
    """
    test_dir = Path("/fsx/mmts/raw/val")
    
    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist")
        return
    
    # Get all sample directories (numeric folders)
    sample_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        sub_dir = sample_dir / "0"
        
        # Skip if subdirectory already exists
        if sub_dir.exists():
            print(f"Subdirectory {sub_dir} already exists, skipping")
            continue
        
        # Create the subdirectory
        sub_dir.mkdir(exist_ok=True)
        
        # Move all files from sample_dir to sub_dir
        files_to_move = [f for f in sample_dir.iterdir() if f.is_file()]
        
        for file_path in files_to_move:
            dest_path = sub_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
        
        print(f"Created {sub_dir} and moved {len(files_to_move)} files")
    
    print("Directory restructuring completed")

if __name__ == "__main__":
    create_subdirectories()