import os
import subprocess

patterns = [
    "__MACOSX", 
    ".DS_Store", 
    ".ipynb_checkpoints", 
    "*.pyc", 
    "__pycache__",
]

def clean_junk_files(root="."):
    abs_root = os.path.abspath(root)
    if os.path.exists(abs_root):
        print(f"[INFO] Cleaning in directory: {abs_root}")

        for pattern in patterns:
            cmd = f"find {abs_root} -iname '{pattern}' -exec rm -rf {{}} +"
            print(f"[CMD] {cmd}")
            subprocess.run(cmd, shell=True)
    else:
        print(f"[INFO] Directory does not exist: {abs_root}")


wandb_patterns = [
    "*/media/images",
]

def clean_wandb_local_vis(root="."):
    abs_root = os.path.abspath(root)
    if os.path.exists(abs_root):
        print(f"[INFO] Cleaning in directory: {abs_root}")
        
        for pattern in wandb_patterns:
            cmd = f"find {abs_root} -type d -path '{pattern}' -exec rm -rf {{}} +"
            print(f"[CMD] {cmd}")
            subprocess.run(cmd, shell=True)
    else:
        print(f"[INFO] Directory does not exist: {abs_root}")

if __name__ == "__main__":
    clean_junk_files()
    clean_wandb_local_vis()