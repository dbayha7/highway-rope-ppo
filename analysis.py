# analysis.py
import re, glob
import pandas as pd


def load_and_parse(pattern="artifacts/combined_validated_data*.csv"):
    path = glob.glob(pattern)[0]
    records = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            tokens = line.rstrip("\n").split(",")
            # find split point where "_epochs=" appears
            idx = next((i for i, t in enumerate(tokens) if "_epochs=" in t), None)
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

    # extract exactly the five sweep axes
    rx = re.compile(
        r"^feat=(?P<features>.+)"
        r"_epochs=(?P<epochs>\d+)"
        r"_lr=(?P<lr>[0-9.eE+-]+)"
        r"_hidden_dim=(?P<hidden_dim>\d+)"
        r"_batch_size=(?P<batch_size>\d+)$"
    )
    hyp = df["experiment"].str.extract(rx)
    hyp = hyp.astype(
        {
            "epochs": "int",
            "lr": "float",
            "hidden_dim": "int",
            "batch_size": "int",
        }
    )
    df = pd.concat([df, hyp], axis=1)
    return df


def main():
    df = load_and_parse()
    print("\n=== Overall metrics ===")
    print(df[["final_reward", "max_reward", "training_steps"]].describe())

    print("\n=== By features ===")
    print(df.groupby("features")["final_reward"].agg(["mean", "std", "count"]))

    print("\n=== By learning rate ===")
    print(df.groupby("lr")["final_reward"].agg(["mean", "std", "count"]))

    print("\n=== By epochs/update ===")
    print(df.groupby("epochs")["final_reward"].agg(["mean", "std", "count"]))

    print("\n=== By hidden_dim ===")
    print(df.groupby("hidden_dim")["final_reward"].agg(["mean", "std", "count"]))

    print("\n=== By batch_size ===")
    print(df.groupby("batch_size")["final_reward"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
