import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import multiprocessing
import matplotlib.pyplot as plt
from data_utils import build_ppt_datasets
import argparse

"""
BE SURE TO CHANGE THE RESULTS FOLDER PATH TO MATCH YOUR LOCAL SET UP!
its an absolute path for me rn beacuse it was giving me saving permission errors otherwise and I haven't gotten around to sorting it out

"""

def set_seed(seed):
    """Set the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the Deep MLP model
class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(outputs, labels):
    mae = torch.mean(torch.abs(outputs - labels)).item()
    return mae

def train_mlp(config):
    """
    Train the MLP model according to the provided configuration.
    Saves results and training curves to the specified results directory.
    """
    set_seed(config["seed"])

    # Set device to the specified GPU if available, otherwise CPU
    if config["gpu_device"] is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['gpu_device']}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = config["input_data"].shape[1]
    output_size = config["labels"].shape[1]

    # Initialize the model
    model_class = config["model"]
    model = model_class(input_size, config["hidden_size"], output_size)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["l2_lambda"])

    # Convert input data and validation data to PyTorch tensors and move to device
    input_tensor = torch.FloatTensor(config["input_data"]).to(device)
    labels_tensor = torch.FloatTensor(config["labels"]).to(device)
    val_tensor = torch.FloatTensor(config["val_data"]).to(device)
    val_labels_tensor = torch.FloatTensor(config["val_labels"]).to(device)

    # Lists to store training and validation losses and MAE
    training_losses = []
    validation_losses = []
    training_mae = []
    validation_mae = []

    # Training loop
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        outputs = model(input_tensor)
        loss = criterion(outputs, labels_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training loss and MAE
        training_losses.append(loss.item())
        train_mae = calculate_mae(outputs, labels_tensor)
        training_mae.append(train_mae)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_labels_tensor)
            val_mae = calculate_mae(val_outputs, val_labels_tensor)
            validation_losses.append(val_loss.item())
            validation_mae.append(val_mae)

        # Print progress
        if (epoch + 1) % config["print_step"] == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}] '
                  f'Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}, '
                  f'Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}')
    
    # Evaluate on test data if provided
    test_loss = None
    test_mae = None
    if "test_data" in config and "test_labels" in config:
        test_tensor = torch.FloatTensor(config["test_data"]).to(device)
        test_labels_tensor = torch.FloatTensor(config["test_labels"]).to(device)
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_tensor)
            test_loss = criterion(test_outputs, test_labels_tensor).item()
            test_mae = calculate_mae(test_outputs, test_labels_tensor)
        print(f'Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}')

    # Save results and plots
    save_results_and_plots(config, model, training_losses, validation_losses,
                           training_mae, validation_mae, test_loss, test_mae)




def convert_to_serializable(obj):
    """Recursively convert non-serializable objects (like numpy arrays) into serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (dict, list, tuple)):
        # Recursively convert elements of dictionaries, lists, and tuples
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj  # These types are directly serializable
    else:
        # If the object is not serializable, return its string representation
        return str(obj)


def save_results_and_plots(config, model, training_losses, validation_losses,
                           training_mae, validation_mae, test_loss, test_mae):
    """Save training results and plots to a dedicated directory for the experiment."""
    
    # Create a base results directory if it doesn't exist
    base_results_dir = config["results_dir"]
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)


    experiment_dir = os.path.join(base_results_dir, config['experiment_name'])

    # Create the experiment-specific directory
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Prepare data to save in a JSON file
    save_data = {
        "config": config,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "training_mae": training_mae,
        "validation_mae": validation_mae,
        "test_loss": test_loss,
        "test_mae": test_mae
    }

    # Convert data to serializable format
    serializable_data = convert_to_serializable(save_data)

    # Save the results to a JSON file inside the experiment directory
    result_filename = os.path.join(experiment_dir, f"results.json")
    with open(result_filename, 'w') as f:
        json.dump(serializable_data, f)

    # Plot and save the training and validation loss curves
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, f"loss_curve.png"))
    plt.close()

    # Plot and save the training and validation MAE curves
    plt.figure()
    plt.plot(training_mae, label='Training MAE')
    plt.plot(validation_mae, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE Curves')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, f"mae_curve.png"))
    plt.close()

    print(f"Results and plots saved to {experiment_dir}")


def perform_grid_search(configs, gpu_ids=None):
    """
    Perform a grid search over the configurations, assigning each experiment to available GPUs.
    """
    processes = []
    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        num_gpus = len(gpu_ids)
    print(f"Number of GPUs available: {num_gpus}, GPU IDs: {gpu_ids}")

    for i, config in enumerate(configs):
        gpu_id = gpu_ids[i % num_gpus]
        config["gpu_device"] = gpu_id
        config["exp_id"] = i  # Assign an experiment ID
        p = multiprocessing.Process(target=train_mlp, args=(config,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run grid search for MLP models.")
    parser.add_argument("--available_gpu_ids", type=int, nargs='+', default=[0, 1, 2, 3],
                        help="List of available GPU IDs (default: [0, 1, 2, 3])")
    parser.add_argument("--results_dir", type=str, default="/data1/home/kivelsons/biogrok/results/by_timepoint",
                        help="Directory to save the results (default: /data1/home/kivelsons/biogrok/results/by_timepoint)")
    parser.add_argument("--datadir", type=str, default="../data",
                        help="Directory containing the CSV data files (default: ../data)")
    parser.add_argument("--n_datapoints", type=str, default=5000, help="number of datapoints to use. use all if you want all available data used")

    return parser.parse_args()

if __name__ == "__main__":

    """
    FOR NOW DOING MOST OF THE PARAM SETTING FOR THE GRID SEARCH MANUALLY IN HERE CUZ ITS EASIER FOR ME TO LOOK AT

    ALSO NOTE THIS IS MULTIPROCESSED SO YOUR PRINT STATEMENTS MIGHT BE INTERLEVED FOR DIFF PARAM SETS
    
    """

    args = parse_arguments()

    # get datasets
    csv_files = os.listdir(args.datadir)
    filenames = [os.path.join(args.datadir, f) for f in csv_files]
    datasets = build_ppt_datasets(filenames, args.n_datapoints)

    # Define your configurations for the grid search
    configs = []
    exp_counter = 0
    for name, model in [("mlp", MLP), ("deep_mlp", DeepMLP)]:
        for hidden_size in [512]:
            for l2_lambda in [0.0]:
                for learning_rate in [0.0001]:
                    config = {
                        "input_data": datasets["train"]["inputs"],
                        "labels": datasets["train"]["labels"],
                        "val_data": datasets["val"]["inputs"],
                        "val_labels": datasets["val"]["labels"],
                        "test_data": datasets["test"]["inputs"],
                        "test_labels": datasets["test"]["labels"],
                        "model": model,
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "epochs": 100000,
                        "seed": 42,
                        "gpu_device": None,  # Will be set in perform_grid_search
                        "l2_lambda": l2_lambda,
                        "print_step": 2000,
                        "results_dir": args.results_dir,
                        "experiment_name": f'{name}_hs_{hidden_size}_lr_{learning_rate}_wd_{l2_lambda}_datasz_{datasets["train"]["inputs"].shape[0]}',  # Will be generated if None
                        "exp_id": exp_counter,
                    }
                configs.append(config)
                exp_counter += 1

    # List of available GPU IDs - for now just "nvidia-smi ing" it to get the list before running
    # kinda hacky but works for now
    #TODO maybe make more things command line params, its just easier for me to look at this way
    
    available_gpu_ids = [0, 1, 2, 3]

    # Run grid search with specified GPU IDs
    perform_grid_search(configs, gpu_ids=available_gpu_ids)
