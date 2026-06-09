import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_and_preprocess_data


class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_baseline_mlp(num_trials=10, epochs=20, lr=0.001, save_dir="saved_weights"):
    bundle = load_and_preprocess_data()
    input_dim = bundle["X_processed"].shape[1]

    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_baseline_mlp.pth")

    best_accuracy = 0.0
    print(f"Starting {num_trials} training runs to find best weights...\n")

    for trial in range(num_trials):
        print(f"--- Trial {trial + 1}/{num_trials} ---")

        model = BaselineMLP(input_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for data, target in bundle["train_loader"]:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        print(f"Training Loss: {epoch_loss / max(1, len(bundle['train_loader'])):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in bundle["test_loader"]:
                outputs = model(data)
                predicted = (outputs >= 0).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_accuracy = 100 * correct / max(1, total)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            print(f"best performance : ({best_accuracy:.2f}% -> {test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Weights saved to '{best_model_path}'.")
        print("-" * 40)

    print("=========================================")
    print(f"All runs completed. Final best accuracy: {best_accuracy:.2f}%")
    return best_model_path, best_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline MLP on the preprocessed credit dataset")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-dir", type=str, default="saved_weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline_mlp(num_trials=args.num_trials, epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)
