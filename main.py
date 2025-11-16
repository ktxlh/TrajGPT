import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from modules.model import TrajGPT
from utils.constants import *
from utils.dataset import collate_fn
from utils.metrics import *
from utils.parser import parse_args
from utils.preprocess import *


def train(step):
    """
    Trains the model using the provided optimizer and training data.

    Returns:
        list: A list of float. Each is the loss of an iteration.
    """
    model.train()

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input, target = convert_batch_to_model_io(batch, args.device)

        output = model(input)
        loss = compute_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        run.log({"train/loss": loss.item()}, step=step)
        pbar.set_postfix({"loss": loss.item()})
    pbar.close()
    return step


def evaluate(loader, step):
    """
    Evaluate the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
        loss_prefix (str): The prefix to use for the loss key in the results dictionary.

    Returns:
        list: A list of dictionaries containing the evaluation results for each batch.
    """
    model.eval()

    losses = []
    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        input, target = convert_batch_to_model_io(batch, args.device)

        with torch.no_grad():
            output = model(input)

        loss = compute_loss(output, target)
        losses.append(loss.item())

        pbar.set_postfix({"loss": loss.item()})
    pbar.close()

    avg_loss = sum(losses) / len(losses)
    run.log({f"val/loss": avg_loss}, step=step)
    return avg_loss


def train_all_epochs():
    """Train model with early stopping."""
    step = 0
    best_loss = float("inf")
    for epoch in range(args.max_num_epochs):
        print(f"Epoch {epoch+1}/{args.max_num_epochs}")
        step = train(step)
        val_loss = evaluate(val_loader, step)

        # Copy the state_dict of the best model
        if val_loss < best_loss:
            best_loss = val_loss

            # Save model and embeddings
            torch.save(model.state_dict(), f"output/{run.name}_model.pt")

            embeddings = {
                idx: model.input.place_embedding.weight.data[idx].cpu()
                for idx in range(model.input.place_embedding.num_embeddings)
            }
            torch.save(embeddings, f"output/{run.name}.pt")


def set_seed():
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If running on CUDA, seed all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """DataLoader worker seeding: ensures each worker has a distinct deterministic seed"""
    # each worker gets seed = base_seed + worker_id
    worker_seed = args.seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed()

    run = wandb.init(
        project="me-poi-baselines",
        name=f"trajgpt_{args.city}_lr{args.lr}",
        config=vars(args),
    )

    (train_data, val_data, num_places, lambda_max) = load_dataset(
        args.city, args.train_ratio
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    model = TrajGPT(num_places, RAW_SEQ_LEN, lambda_max).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_all_epochs()

    run.finish()
