import argparse
import pandas as pd
import re, os


def calculate_fid_bias(success_csv, epsilons=[0.02, 0.05, 0.1, 0.15, 0.2]):
    with open(success_csv, 'r') as f:
        lines = f.readlines()

    datasets = {}
    current_dataset = None

    # Parse file line-by-line
    for line in lines:
        line = line.strip()
        if line.startswith("# ---"):
            current_dataset = re.findall(r'# --- (.+?) ---', line)[0].strip()
            datasets[current_dataset] = []
        elif line and not line.startswith('#'):
            datasets[current_dataset].append(line)

    bias_rows = []
    dataset_count = {eps: 0 for eps in epsilons}

    total_bias = {"Dataset": "Total MEAN"}
    for eps in epsilons:
        total_bias[f"ε={eps}"]=0
    # print(total_bias)

    for dataset, data in datasets.items():
        if len(data) < 2:
            continue

        df = pd.DataFrame([row.split(',') for row in data[1:]], columns=data[0].split(','))
        df.replace("", pd.NA, inplace=True)
        demographics = [col for col in df.columns if col not in {"Dataset", "MEAN_PER_DATASET", "MEAN_ALL", "MEAN", "NaN"}]
        df[demographics] = df[demographics].apply(pd.to_numeric, errors='coerce')

        count_dataset = {eps: 0 for eps in epsilons}
        count_eop_dataset = {eps: 0 for eps in epsilons}

        for _, row in df.iterrows():
            values = row[demographics].dropna().to_dict()
            demos = list(values.keys())
            for demo in demos:
                row_entry = {"Dataset": dataset, "Demographic": demo}
                total = len(demos) - 1
                for eps in epsilons:
                    count = sum(values[demo] - values[other_demo] < -eps for other_demo in demos if other_demo != demo)
                    bias_percent = int(round((count / total) * 100))
                    row_entry[f"ε={eps}"] = bias_percent
                    count_dataset[eps] += total
                    count_eop_dataset[eps] += count
                bias_rows.append(row_entry)

        total_pairs = len(demographics) * (len(demographics) - 1)/2
        bias_dataset = {"Dataset": dataset, "Demographic": "MEAN"}
        # print(total_bias)
        for eps in epsilons:
            mean_bias = int(round((count_eop_dataset[eps] / total_pairs) * 100)) if total_pairs else 0
            bias_dataset[f"ε={eps}"] = mean_bias
            total_bias[f"ε={eps}"]+=mean_bias
            dataset_count[eps]+=1

        bias_rows.append(bias_dataset)

    for eps in epsilons:
        total_bias[f"ε={eps}"]/=dataset_count[eps]
        total_bias[f"ε={eps}"]= int(round(total_bias[f"ε={eps}"]))
    bias_rows.append(total_bias)
    return pd.DataFrame(bias_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute demographic bias from FID success scores.")
    parser.add_argument("--fid_success", type=str, required=True, help="Path to fid_success_<method>.csv")
    args = parser.parse_args()

    out_df = calculate_fid_bias(args.fid_success)
    method = os.path.basename(args.fid_success).replace("fid_success_", "").replace(".csv", "")
    fol='bias'
    if not os.path.isdir(fol):
        os.mkdir(fol)
    out_path = fol + '/' + f"bias_{method}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved bias results to {out_path}")
