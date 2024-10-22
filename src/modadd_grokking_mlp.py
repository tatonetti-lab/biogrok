import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import einops
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_mae(outputs, labels):
    mae = torch.mean(torch.abs(outputs - labels)).item()
    return mae

class DeepMLP(nn.Module):
    def __init__(self, p, hidden_size=24):
        super(DeepMLP, self).__init__()
        self.p = p
        self.hidden_size = hidden_size
        self.W_input = nn.Linear(p, hidden_size, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.W_output = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        a = x[:, 0].long()
        b = x[:, 1].long()
        a_onehot = F.one_hot(a, num_classes=self.p).float()
        b_onehot = F.one_hot(b, num_classes=self.p).float()
        hidden = self.relu1(self.W_input(a_onehot) + self.W_input(b_onehot))
        hidden = self.fc2(hidden)
        hidden = self.relu2(hidden)
        output = self.W_output(hidden)
        return output

class MLP(nn.Module):
    def __init__(self, p, hidden_size=24):
        super(MLP, self).__init__()
        self.p = p
        self.hidden_size = hidden_size
        self.W_input = nn.Linear(p, hidden_size, bias=True)
        self.W_output = nn.Linear(hidden_size, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        a = x[:, 0].long()
        b = x[:, 1].long()
        a_onehot = F.one_hot(a, num_classes=self.p).float()
        b_onehot = F.one_hot(b, num_classes=self.p).float()
        hidden = self.relu(self.W_input(a_onehot) + self.W_input(b_onehot))
        output = self.W_output(hidden)
        return output

def train_mlp(args):
    set_seed(args.seed)


     # Create results directory if it doesn't exist
    plot_dir = Path("results/modadd_grok/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")
    print(f"Using device: {device}")

    # Generate data
    a_vector = einops.repeat(torch.arange(args.p), "i -> (i j)", j=args.p)
    b_vector = einops.repeat(torch.arange(args.p), "j -> (i j)", i=args.p)
    dataset = torch.stack([a_vector, b_vector], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % args.p

    # Split data
    torch.manual_seed(args.data_seed)
    indices = torch.randperm(args.p * args.p)
    cutoff = int(args.p * args.p * args.train_frac)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices].reshape(-1, 1)
    test_data = dataset[test_indices]
    test_labels = labels[test_indices].reshape(-1, 1)

    # Convert to float
    train_data = train_data.float()
    test_data = test_data.float()
    train_labels = train_labels.float()
    test_labels = test_labels.float()

    # Initialize model
    if args.depth == "shallow":
        model = MLP(p=args.p, hidden_size=args.hidden_size).to(device)
    else:
        model = DeepMLP(p=args.p, hidden_size=args.hidden_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

    # Training loop
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())
        train_mae = calculate_mae(outputs, train_labels)
        training_accuracies.append(train_mae)

        model.eval()
        with torch.no_grad():
            val_outputs = model(test_data)
            val_loss = criterion(val_outputs, test_labels)
            val_mae = calculate_mae(val_outputs, test_labels)

        validation_losses.append(val_loss.item())
        validation_accuracies.append(val_mae)

        if (epoch + 1) % args.print_step == 0:
            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}, '
                  f'Validation Loss: {val_loss.item():.4f}, Train MAE: {train_mae:.4f}, '
                  f'Val MAE: {val_mae:.4f}')

    # Plot results
    if args.plot:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(training_accuracies, label='Training MAE')
        plt.plot(validation_accuracies, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.title('Training and Validation MAE')

        plt.tight_layout()
        # plt.savefig(f'training_curves_{args.depth}_p{args.p}.png')

        plot_path = plot_dir / f'training_curves_{args.depth}_p{args.p}.png'
        plt.savefig(plot_path)

        if args.show_plot:
            plt.show()

    return {
        "model": model,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "training_acc": training_accuracies,
        "validation_acc": validation_accuracies
    }

def main():
    parser = argparse.ArgumentParser(description='Train MLP for modular arithmetic')
    
    # Data parameters
    parser.add_argument('--p', type=int, default=67, help='Modulo value')
    parser.add_argument('--train_frac', type=float, default=0.2, help='Fraction of data for training')
    parser.add_argument('--data_seed', type=int, default=598, help='Seed for data splitting')

    # Model parameters
    parser.add_argument('--depth', type=str, choices=['shallow', 'deep'], default='deep', help='Model depth')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size')

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--l2_lambda', type=float, default=0.1, help='L2 regularization strength')
    parser.add_argument('--print_step', type=int, default=100, help='Print frequency')

    # Hardware parameters
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID (None for CPU)')

    # Visualization parameters
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--show_plot', action='store_true', help='Show plots (in addition to saving)')

    args = parser.parse_args()
    
    results = train_mlp(args)
    
    # Save final model if needed
    # torch.save(results['model'].state_dict(), f'model_{args.depth}_p{args.p}.pt')

if __name__ == "__main__":
    main()