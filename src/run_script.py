import subprocess
import sys
import os

# The list of scripts to run in specific order
scripts = [
    "collect_images.py",
    "preprocess.py",
    "train_model.py"
]

print("Starting Automation Pipeline...")

# Verify we are in the correct directory
current_dir = os.path.basename(os.getcwd())
if current_dir != "src":
    print("  WARNING: You do not seem to be in the 'src' folder.")
    print(f"   Current folder: {os.getcwd()}")
    print("   Please 'cd src' before running this script.")
    sys.exit(1)

for script in scripts:

    print(f"Running: {script}")


    try:
        # sys.executable ensures it uses the same python version (python3) you are currently using
        result = subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError:
        print(f"\n ERROR: {script} crashed or failed. Pipeline stopped.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n ERROR: Could not find file '{script}'. Are you in the right folder?")
        sys.exit(1)


print("SUCCESS: All scripts finished correctly.")
