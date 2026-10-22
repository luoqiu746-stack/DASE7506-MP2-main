
"""E12: Package BCE and Pairwise MLP into one checkpoint."""

from pathlib import Path
import numpy as np


SOURCE_MODELS = {
    "bce": Path("runs/e6c-mlp.npz"),
    "pair": Path("runs/e10-pairwise.npz"),
}

OUTPUT = Path("runs/e12-ensemble.npz")

REQUIRED = ("w1", "b1", "w2", "b2", "mean", "std")


def main():
    combined = {}

    for prefix, path in SOURCE_MODELS.items():
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as model:
            for key in REQUIRED:
                combined[f"{prefix}_{key}"] = model[key].copy()

        print(f"Loaded {prefix}: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **combined)

    print(f"Saved: {OUTPUT}")
    print(f"Checkpoint size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
