import argparse

from calrep.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("Loaded config:")
    print(cfg)
    print("\nTraining not implemented yet (Issue #3).")


if __name__ == "__main__":
    main()
