import os
import subprocess

def clean_junk_files(root="."):
    abs_root = os.path.abspath(root)
    print(f"[INFO] Cleaning in directory: {abs_root}")
    
    patterns = ["__MACOSX", ".DS_Store", ".ipynb_checkpoints", "*.pyc", "__pycache__"]
    for pattern in patterns:
        cmd = f"find {abs_root} -iname '{pattern}' -exec rm -rf {{}} +"
        print(f"[CMD] {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    clean_junk_files()