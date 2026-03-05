import argparse

from calrep.data import get_cifar_dataloaders
from calrep.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_loader, val_loader, test_loader, meta = get_cifar_dataloaders(cfg)

    print("Loaded dataset:", meta.dataset_name)
    print("Classes:", meta.num_classes)
    print(
        "Sizes:",
        {"train": meta.train_size, "val": meta.val_size, "test": meta.test_size},
    )
    print("Batch size:", train_loader.batch_size)


if __name__ == "__main__":
    main()
