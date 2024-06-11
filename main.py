import torch
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from modules.model import TrajGPT
from utils.constants import *
from utils.metrics import *
from utils.preprocess import *


def train(model, optimizer, train_loader, task, device, num_regions):
    """
    Trains the model using the provided optimizer and training data.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        task (str): The task being performed.
        device (torch.device): The device to be used for training.
        num_regions (int): The number of regions.

    Returns:
        float: The mean loss over all training batches.
    """
    model.train()

    losses = []
    for batch in train_loader:
        input, target = convert_batch_to_model_io(task, batch, device)

        output = model(input)
        loss = compute_loss(output, target, num_regions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    return np.mean(losses)


def evaluate(model, loader, task, device, num_regions, loss_prefix):
    """
    Evaluate the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
        task (str): The task being performed.
        device (torch.device): The device to use for evaluation.
        num_regions (int): The number of regions.
        loss_prefix (str): The prefix to use for the loss key in the results dictionary.

    Returns:
        list: A list of dictionaries containing the evaluation results for each batch.
    """
    model.eval()

    results = []
    for batch in loader:
        input, target = convert_batch_to_model_io(task, batch, device)

        with torch.no_grad():
            output = model(input)
            
        loss = compute_loss(output, target, num_regions)
        scores = compute_scores(input, output, target, task, max_duration, max_travel_time)
        results.append({f'{loss_prefix}_loss': loss.item(), **scores})
    
    result_dict = pd.json_normalize(results).mean().to_dict()
    return result_dict


if __name__ == "__main__":
    # Comment out whichever task you are not running
    # Uncomment the task you are running
    # task = NEXT_PREDICTION
    task = INFILLING

    num_epochs = 10000
    patience = 50  # unit: epoch

    train_data, val_data, test_data, num_regions, lambda_max, max_duration, max_travel_time = load_geolife_dataset(task)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajGPT(num_regions + N_SPECIAL_TOKENS, SEQ_LEN[task], lambda_max).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_epoch, best_score, best_state_dict = -1, float("inf"), None
    train_loss, val_score = [], []
    for epoch in trange(num_epochs):
        train_loss.append(train(model, optimizer, train_loader, task, device, num_regions))
        val_score.append(evaluate(model, val_loader, task, device, num_regions, 'val'))

        # Save the best model
        if val_score[-1]['val_loss'] < best_score:
            best_epoch = epoch
            best_score = val_score[-1]['val_loss']
            best_state_dict = model.state_dict()
        
        # Early stopping
        if epoch - best_epoch >= patience: break

    # Test
    model.load_state_dict(best_state_dict)
    model.to(device)
    test_score = evaluate(model, test_loader, task, device, num_regions, 'test')
    print(test_score)

    # DEBUG
    for k, v in test_score.items():
        if "loss" not in k:
            v *= 100
        print(f"{v:.2f}", end=",")
    
    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.savefig("train_loss.png")
    plt.close()
