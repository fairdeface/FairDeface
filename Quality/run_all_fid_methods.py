import subprocess, os

# List of methods based on your folder names
methods = [
    "DP1",
    "DP2",
    "Fawkes-High",
    "DP-Snow",
    "K-Same-Pixel-ArcFace",

  #  "CIAGAN",
   
    "Pixelation"
]

dataset_root = "../datasets"
batch_size = 150

for method in methods:
    print(f"\n🚀 Running FID for: {method}")
    
    cmd = [
        "python", "FID.py",
        "--method", method,
        "--dataset_root", dataset_root,
        "--batch_size", str(batch_size)
    ]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ Failed: {method}")
    else:
        print(f"✅ Done: {method}")
