# analysis.py
# Self-contained script to aggregate and summarize pilot experiment results

import re
import glob
import pandas as pd


def load_and_parse(pattern="artifacts/combined_validated_data*.csv"):
    # pick whichever combined CSV is present
    path_list = glob.glob(pattern)
    if not path_list:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    path = path_list[0]
    records = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            tokens = line.strip().split(",")
            # find the boundary where "_epochs=" appears in experiment name
            idx = next((i for i, t in enumerate(tokens) if t.startswith("feat=")), None)
            if idx is None or len(tokens) < idx + 4:
                continue
            exp = ",".join(tokens[: idx + 1])
            try:
                final = float(tokens[idx + 1])
                _max = float(tokens[idx + 2])
                steps = int(tokens[idx + 3])
            except ValueError:
                continue
            records.append(
                {
                    "experiment": exp,
                    "final_reward": final,
                    "max_reward": _max,
                    "training_steps": steps,
                }
            )
    df = pd.DataFrame(records)

    # extract hyperparameters via regex
    rx = re.compile(
        r"^feat=(?P<features>[^_]+)"
        r".*_epochs=(?P<epochs>\d+)"
        r".*_lr=(?P<lr>[0-9.eE+-]+)"
        r".*_hidden_dim=(?P<hidden_dim>\d+)"
        r".*_batch_size=(?P<batch_size>\d+)"
        r".*_clip_eps=(?P<clip_eps>[0-9.eE+-]+)"
        r".*_entropy_coef=(?P<entropy_coef>[0-9.eE+-]+)$"
    )
    hyp = df["experiment"].str.extract(rx)
    # cast types
    for c, t in [("epochs", int), ("hidden_dim", int), ("batch_size", int)]:
        if c in hyp:
            hyp[c] = hyp[c].astype(t)
    for c in ["lr", "clip_eps", "entropy_coef"]:
        if c in hyp:
            hyp[c] = hyp[c].astype(float)
    df = pd.concat([df, hyp], axis=1)
    return df


def main():
    df = load_and_parse()
    # Summary tables
    summaries = {
        "Overall metrics": df[
            ["final_reward", "max_reward", "training_steps"]
        ].describe(),
        "By features": df.groupby("features")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By learning rate": df.groupby("lr")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By epochs": df.groupby("epochs")["final_reward"].agg(["mean", "std", "count"]),
        "By hidden_dim": df.groupby("hidden_dim")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By batch_size": df.groupby("batch_size")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By clip_eps": df.groupby("clip_eps")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By entropy_coef": df.groupby("entropy_coef")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
    }
    for title, table in summaries.items():
        print(f"\n=== {title} ===\n{table}\n")


if __name__ == "__main__":
    main()
