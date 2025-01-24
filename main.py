from copy import deepcopy

import pandas as pd
import torch
from torch.optim import Adafactor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from modules.model import TrajGPT
from utils.constants import *
from utils.metrics import *
from utils.parser import parse_args
from utils.preprocess import *


def train():
    """
    Trains the model using the provided optimizer and training data.

    Returns:
        list: A list of float. Each is the loss of an iteration.
    """
    model.train()

    losses = []
    for batch in train_loader:
        input, target = convert_batch_to_model_io(args.task, batch, args.device)

        output = model(input)
        loss = compute_loss(output, target, num_regions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    return losses


def evaluate(loader, loss_prefix):
    """
    Evaluate the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
        loss_prefix (str): The prefix to use for the loss key in the results dictionary.

    Returns:
        list: A list of dictionaries containing the evaluation results for each batch.
    """
    model.eval()

    results = []
    for batch in loader:
        input, target = convert_batch_to_model_io(args.task, batch, args.device)

        with torch.no_grad():
            output = model(input)
            
        loss = compute_loss(output, target, num_regions)
        scores = compute_scores(input, output, target, args.task, max_duration, max_travel_time)
        results.append({f'{loss_prefix}_loss': loss.item(), **scores})
    
    result_dict = pd.json_normalize(results).mean().to_dict()
    return result_dict


def train_model_with_early_stopping():
    """Train model with early stopping."""
    best_epoch, best_score, best_state_dict = -1, float("inf"), None
    train_loss, val_score = [], []
    for epoch in trange(args.max_num_epochs):
        train_loss.extend(train())
        val_score.append(evaluate(val_loader, 'val'))

        # Copy the state_dict of the best model
        if val_score[-1]['val_loss'] < best_score:
            best_epoch = epoch
            best_score = val_score[-1]['val_loss']
            best_state_dict = deepcopy(model.state_dict())
        
        # Early stopping
        if epoch - best_epoch >= args.patience: break
    
    return best_state_dict


def test_model(state_dict):
    """Test the best model given the state_dict."""
    model.load_state_dict(state_dict)
    model.to(args.device)
    test_score = evaluate(test_loader, 'test')
    return test_score


if __name__ == "__main__":
    args = parse_args()

    train_data, val_data, test_data, num_regions, lambda_max, max_duration, max_travel_time = load_geolife_dataset(args.task)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    model = TrajGPT(num_regions + N_SPECIAL_TOKENS, SEQ_LEN[args.task], lambda_max).to(args.device)
    optimizer = Adafactor(model.parameters())

    best_state_dict = train_model_with_early_stopping()    
    test_score = test_model(best_state_dict)
    print(test_score)
