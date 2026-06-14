"""
Train and evaluate one selected CNN model variant
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_dataset
from models import get_model

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def prepare_tensors(path):
    """
    Loading one processed dataset and convert its arrays into tensors (pytorch) 
    The function returns sequence tensor, activity target tensor and cell-type index tensor
    """
    _, x, y, cell_type = load_dataset(path)

    # DNA sequences and activity values must use floats
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # embedding layers require torch.long format
    cell_type = torch.tensor(cell_type, dtype=torch.long)

    return x, y, cell_type


def evaluate(model, loader, loss_fn, device):
    """
    Evaluating (without parameter updates)
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    # disabling gradients 
    with torch.no_grad():
        for x_batch, cell_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            cell_batch = cell_batch.to(device)
            y_batch = y_batch.to(device)
            
            # forward through a selected model
            pred = model(x_batch, cell_batch)
            
            # MSE between predictions vs true activity
            loss = loss_fn(pred, y_batch)

            # storing the loss for further steps
            total_loss += loss.item()
            
            predictions.append(pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

        # averaging the mean loss across all batches
        avg_loss = total_loss / len(loader)
        
        # combining arrays from individual batches into ona array
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        return avg_loss, predictions, targets
    

def train(args):
    """
    Training one model variant and saving its outputs
    """
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    x_train, y_train, cell_train = prepare_tensors(train_path)
    x_val, y_val, cell_val = prepare_tensors(val_path)
    x_test, y_test, cell_test = prepare_tensors(test_path)

    train_loader = DataLoader(TensorDataset(x_train, cell_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, cell_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, cell_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(args.variant).to(device)
    
    # Adam updates the model parameters using calculated gradients
    # learning rate controls the size of parameter updates
    # weight decay provides regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # mean squared error used as the regression loss for the models
    loss_fn = nn.MSELoss()

    # storing training and validation loss after every epoch
    history = []

    # train run
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x_batch, cell_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            cell_batch = cell_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            pred = model(x_batch, cell_batch)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # calculating the average training loss across all batches
        train_loss /= len(train_loader)

        # evaluating the curent model on validation
        # returned predictions and targets are ignored here since only val loss is needed during training
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)

        # save loss values for further plotting of the training curves
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
        )

    # final evaluation on the test set
    test_loss, test_predictions, test_targets = evaluate(model, test_loader, loss_fn, device)

    # save model, results, MSE
    torch.save(model.state_dict(), MODELS_DIR / f"{args.variant}.pt")

    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{args.variant}_history.csv", index=False)
    pd.DataFrame(
        {
            "target": test_targets,
            "prediction": test_predictions
        }
    ).to_csv(RESULTS_DIR / f"{args.variant}_test_results.csv", index=False)

    pd.DataFrame(
        [
            {
                "variant": args.variant,
                "test_mse": test_loss
            }
        ]
    ).to_csv(RESULTS_DIR / f"{args.variant}_test_metrics.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for gene expression prediction.")

    # choosing architecure
    parser.add_argument(
        "--variant",
        choices=["baseline", "embedding", "attention", "full"],
        required=True,
        help="Model variant to train."
    )

    # number of epochs, batch size, learning rate and weight decay can be changes
    # default values performed the best during trials
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)