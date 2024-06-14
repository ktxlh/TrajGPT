import argparse
from utils.constants import INFILLING, NEXT_PREDICTION

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Train for one of `infilling` or `next_prediction`", choices=["infilling", "next_prediction"], required=True)
    parser.add_argument("--max_num_epochs", help="Maximum number of epochs of model training", type=int, default=2000)
    parser.add_argument("--patience", help="Patience for early stopping (unit: epoch)", type=int, default=50)
    parser.add_argument("--lr", help="Learning rate for training", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", help="Batch size for training models", type=int, default=64)
    parser.add_argument("--test_batch_size", help="Batch size for testing models", type=int, default=256)
    parser.add_argument("--device", help="Device to use for training and testing models", type=str, default="cuda")
    args = parser.parse_args()

    args.task = INFILLING if args.task == 'infilling' else NEXT_PREDICTION
    assert (args.device == "cpu") or ("cuda" in args.device), "Please specify a valid device (cpu or cuda)"
    return args
