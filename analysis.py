# analysis.py
import re
import pandas as pd
import glob


def load_and_parse(path_pattern):
    records = []
    pattern = re.compile(
        r"^feat=(?P<features>.+)_epochs=(?P<epochs>\d+)_lr=(?P<lr>[0-9.eE+-]+)"
        r"_hidden_dim=(?P<hidden_dim>\d+)_batch_size=(?P<batch_size>\d+)$"
    )
    # find whichever combined_validated_data file is present
    path = glob.glob(path_pattern)[0]
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            tokens = line.strip().split(",")
            idx = next((i for i, t in enumerate(tokens) if "_epochs=" in t), None)
            if idx is None or len(tokens) < idx + 4:
                continue
            exp = ",".join(tokens[: idx + 1])
            try:
                final = float(tokens[idx + 1])
                mx = float(tokens[idx + 2])
                steps = int(tokens[idx + 3])
            except ValueError:
                continue
            records.append(
                {
                    "experiment": exp,
                    "final_reward": final,
                    "max_reward": mx,
                    "training_steps": steps,
                }
            )
    df = pd.DataFrame(records)
    hyp = df["experiment"].str.extract(pattern)
    hyp = hyp.astype(
        {"epochs": "int", "lr": "float", "hidden_dim": "int", "batch_size": "int"}
    )
    hyp["features"] = hyp["features"]
    return pd.concat([df, hyp], axis=1)


def main():
    df = load_and_parse("artifacts/combined_validated_data*.csv")

    summaries = {
        "Overall": df[["final_reward", "max_reward", "training_steps"]].describe(),
        "By features": df.groupby("features")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By lr": df.groupby("lr")["final_reward"].agg(["mean", "std", "count"]),
        "By epochs": df.groupby("epochs")["final_reward"].agg(["mean", "std", "count"]),
        "By hidden": df.groupby("hidden_dim")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
        "By batch": df.groupby("batch_size")["final_reward"].agg(
            ["mean", "std", "count"]
        ),
    }
    for title, table in summaries.items():
        print(f"\n=== {title} ===\n{table}\n")


if __name__ == "__main__":
    main()
