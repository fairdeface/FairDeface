import os
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from collections import defaultdict

# Argument parsing
parser = argparse.ArgumentParser(description="Calculate FID and normalized FID-based success per demographic.")
parser.add_argument('--method', type=str, required=True, help="Name of the obfuscation method (e.g., DP1, CIAGAN)")
parser.add_argument('--dataset_root', type=str, default="../datasets", help="Root folder containing all datasets")
parser.add_argument('--batch_size', type=int, default=50, help="Batch size for FID computation (default: 50)")
args = parser.parse_args()

# Derived paths
obfuscated_root = os.path.join(args.dataset_root, args.method)
original_root = os.path.join(args.dataset_root, "Original")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.PILToTensor()
])

def collect_image_pairs(dataset, demographic):
    image_pairs = []

    if args.method.lower() == "ciagan":
        demo_path = os.path.join(obfuscated_root, dataset, demographic)
        if not os.path.exists(demo_path):
            return image_pairs
    
        for identity in os.listdir(demo_path):
            id_path = os.path.join(demo_path, identity)
            if not os.path.isdir(id_path):
                continue
    
            files = os.listdir(id_path)
            org_files = {}
            obf_files = {}
    
            for f in files:
                if f.endswith("_org.jpg"):
                    key = f.replace("_org.jpg", "")
                    org_files[key] = os.path.join(id_path, f)
                elif f.endswith("_obf.jpg"):
                    key = f.replace("_obf.jpg", "")
                    obf_files[key] = os.path.join(id_path, f)
    
            common_keys = set(org_files.keys()) & set(obf_files.keys())
    
            if not common_keys:
                print(f"⚠️ No matches in {id_path}")
            else:
                for key in common_keys:
                    image_pairs.append((org_files[key], obf_files[key]))

    else:
        obf_demo_path = os.path.join(obfuscated_root, dataset, demographic)
        orig_demo_path = os.path.join(original_root, dataset, demographic)
        if not os.path.exists(orig_demo_path):
            return image_pairs

        for identity in os.listdir(obf_demo_path):
            obf_id_path = os.path.join(obf_demo_path, identity)
            orig_id_path = os.path.join(orig_demo_path, identity)

            if not os.path.isdir(obf_id_path) or not os.path.exists(orig_id_path):
                continue

            for img_name in os.listdir(obf_id_path):
                obf_img_path = os.path.join(obf_id_path, img_name)
                orig_img_path = os.path.join(orig_id_path, img_name)
                if os.path.exists(orig_img_path):
                    image_pairs.append((orig_img_path, obf_img_path))

    return image_pairs


def load_images(image_paths):
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(transform(img).unsqueeze(0))
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
    return torch.cat(images) if images else None

def update_in_batches(fid_metric, images, is_real, batch_size, device):
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        fid_metric.update(batch, real=is_real)

# Main FID computation
fid_scores = defaultdict(dict)

try:
    for dataset in os.listdir(obfuscated_root):
        dataset_path = os.path.join(obfuscated_root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for demographic in os.listdir(dataset_path):
            demo_path = os.path.join(dataset_path, demographic)
            if not os.path.isdir(demo_path):
                continue

            image_pairs = collect_image_pairs(dataset, demographic)
            print(f"🔍 {dataset}/{demographic} - Collected {len(image_pairs)} image pairs.")

            if not image_pairs:
                continue

            orig_paths, obf_paths = zip(*image_pairs)
            orig_imgs = load_images(orig_paths)
            obf_imgs = load_images(obf_paths)

            if orig_imgs is None or obf_imgs is None:
                continue

            try:
                fid = FrechetInceptionDistance(feature=2048).to(device)
                update_in_batches(fid, orig_imgs, is_real=True, batch_size=args.batch_size, device=device)
                update_in_batches(fid, obf_imgs, is_real=False, batch_size=args.batch_size, device=device)
                score = fid.compute().item()
                print(f"✅ FID for {dataset} / {demographic}: {score:.2f}")
                fid_scores[dataset][demographic] = score

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️ CUDA OOM for {dataset}/{demographic}. Retrying on CPU...")
                    torch.cuda.empty_cache()
                    cpu_device = torch.device("cpu")
                    fid = FrechetInceptionDistance(feature=2048).to(cpu_device)
                    update_in_batches(fid, orig_imgs.to(cpu_device), is_real=True, batch_size=args.batch_size, device=cpu_device)
                    update_in_batches(fid, obf_imgs.to(cpu_device), is_real=False, batch_size=args.batch_size, device=cpu_device)
                    score = fid.compute().item()
                    print(f"✅ [CPU fallback] FID for {dataset} / {demographic}: {score:.2f}")
                    fid_scores[dataset][demographic] = score
                else:
                    print(f"❌ RuntimeError for {dataset}/{demographic}: {e}")
            finally:
                del fid
                torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("⛔ Interrupted by user. Partial results will be saved.")

# ---- Output one table per dataset to CSV ----
fol1='FIDs'
fol2='FID_success'
if not os.path.isdir(fol1):
    os.mkdir(fol1)
    
if not os.path.isdir(fol2):
    os.mkdir(fol2)

fid_out_path = fol1 + '/' + f"fid_scores_{args.method}.csv"
success_out_path = fol2 + '/' + f"fid_success_{args.method}.csv"

all_fid_values = []
all_success_values = []

with open(fid_out_path, "w") as fid_csv, open(success_out_path, "w") as success_csv:
    for dataset, demo_scores in fid_scores.items():
        # --- FID Table ---
        fid_header = ["Dataset"] + list(demo_scores.keys()) + ["MEAN"]
        fid_values = [round(demo_scores[demo], 2) for demo in demo_scores]
        fid_mean = round(sum(fid_values) / len(fid_values), 2)
        fid_row = [dataset] + fid_values + [fid_mean]

        # Write FID
        fid_csv.write(f"# --- {dataset} ---\n")
        fid_csv.write(",".join(fid_header) + "\n")
        fid_csv.write(",".join(map(str, fid_row)) + "\n\n")

        # --- Success Table ---
        max_fid = max(demo_scores.values()) if demo_scores else 1
        success_header = ["Dataset"] + list(demo_scores.keys()) + ["MEAN"]
        success_values = [round(1 - min(fid, max_fid) / max_fid, 2) for fid in fid_values]
        success_mean = round(sum(success_values) / len(success_values), 2)
        success_row = [dataset] + success_values + [success_mean]

        # Write Success
        success_csv.write(f"# --- {dataset} ---\n")
        success_csv.write(",".join(success_header) + "\n")
        success_csv.write(",".join(map(str, success_row)) + "\n\n")

        all_fid_values.extend(fid_values)
        all_success_values.extend(success_values)

    # Append global mean row
    fid_csv.write("# --- Overall Mean ---\n")
    fid_csv.write("Dataset,MEAN\n")
    fid_csv.write(f"MEAN_ALL,{round(sum(all_fid_values)/len(all_fid_values), 2)}\n")

    success_csv.write("# --- Overall Mean ---\n")
    success_csv.write("Dataset,MEAN\n")
    success_csv.write(f"MEAN_ALL,{round(sum(all_success_values)/len(all_success_values), 2)}\n")

print(f"📄 Saved one-table-per-dataset FID to {fid_out_path}")
print(f"📄 Saved one-table-per-dataset normalized success to {success_out_path}")
