import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        help="Train for one of `infilling` or `next_prediction`",
        choices=["infilling", "next_prediction"],
        default="next_prediction",
    )
    parser.add_argument(
        "--max_num_epochs",
        help="Maximum number of epochs of model training",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr", help="Learning rate for training", type=float, default=0.001
    )
    parser.add_argument(
        "--train_batch_size",
        help="Batch size for training models",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--test_batch_size", help="Batch size for testing models", type=int, default=256
    )
    parser.add_argument(
        "--device",
        help="Device to use for training and testing models",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--seed", help="Random seed for reproducibility", type=int, default=42
    )
    parser.add_argument("--city")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    assert (args.device == "cpu") or (
        "cuda" in args.device
    ), "Please specify a valid device (cpu or cuda)"
    return args
