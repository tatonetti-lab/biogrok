import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import multiprocessing
import matplotlib.pyplot as plt
from data_utils import get_dataloaders
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

    # Get data loaders from config
    trainloader = config["trainloader"]
    valloader = config["valloader"]
    testloader = config["testloader"]

    # Determine input and output sizes from the first batch
    sample_batch, sample_labels = next(iter(trainloader))
    input_size = sample_batch.shape[1]
    output_size = sample_labels.shape[1]

    # Initialize the model
    model_class = config["model"]
    model = model_class(input_size, config["hidden_size"], output_size)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["l2_lambda"])

    # Lists to store training and validation losses and MAE
    training_losses = []
    validation_losses = []
    training_mae = []
    validation_mae = []

    # Training loop
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        for batch, labels in trainloader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mae += calculate_mae(outputs, labels)
        
        avg_loss = epoch_loss / len(trainloader)
        avg_mae = epoch_mae / len(trainloader)
        training_losses.append(avg_loss)
        training_mae.append(avg_mae)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch, labels in valloader:
                batch, labels = batch.to(device), labels.to(device)
                outputs = model(batch)
                val_loss += criterion(outputs, labels).item()
                val_mae += calculate_mae(outputs, labels)
        
        avg_val_loss = val_loss / len(valloader)
        avg_val_mae = val_mae / len(valloader)
        validation_losses.append(avg_val_loss)
        validation_mae.append(avg_val_mae)

        # Print progress
        if (epoch + 1) % config["print_step"] == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}] '
                  f'Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, '
                  f'Train MAE: {avg_mae:.6f}, Val MAE: {avg_val_mae:.6f}')
    
    # Evaluate on test data
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for batch, labels in testloader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            test_loss += criterion(outputs, labels).item()
            test_mae += calculate_mae(outputs, labels)
    
    avg_test_loss = test_loss / len(testloader)
    avg_test_mae = test_mae / len(testloader)
    print(f'Test Loss: {avg_test_loss:.6f}, Test MAE: {avg_test_mae:.6f}')

    # Save results and plots
    save_results_and_plots(config, model, training_losses, validation_losses,
                           training_mae, validation_mae, avg_test_loss, avg_test_mae)




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

     # Plot Prey and Predator predictions vs actuals (Test Data)
    model.eval()
    model.to(config['gpu_device'])
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for inputs, labels in config['testloader']:
            inputs = inputs.to(config['gpu_device'])
            labels = labels.to(config['gpu_device'])
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_actuals.append(labels.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)

    # Prey Predictions vs. Actuals
    plt.figure()
    plt.plot(predictions[:, 0], label='Predicted Prey', alpha=0.7)
    plt.plot(actuals[:, 0], label='Actual Prey', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Prey Population')
    plt.title('Prey Population: Predictions vs. Actuals')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "prey_population.png"))
    plt.close()

    # Predator Predictions vs. Actuals
    plt.figure()
    plt.plot(predictions[:, 1], label='Predicted Predator', alpha=0.7)
    plt.plot(actuals[:, 1], label='Actual Predator', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Predator Population')
    plt.title('Predator Population: Predictions vs. Actuals')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "predator_population.png"))
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
    trainloader, valloader, testloader = get_dataloaders(train_start_end=[0, 200], val_start_end=[200, 300], test_start_end=[300, 400], step_size=1.0, batch_size=32)

    # Define your configurations for the grid search
    configs = []
    exp_counter = 0
    for name, model in [("mlp", MLP), ("deep_mlp", DeepMLP)]:
        for hidden_size in [512]:
            for l2_lambda in [0.0]:
                for learning_rate in [0.0001]:
                    config = {
                        "trainloader": trainloader,
                        "valloader": valloader,
                        "testloader": testloader,
                        "model": model,
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "epochs": 100,
                        "seed": 42,
                        "gpu_device": None,  # Will be set in perform_grid_search
                        "l2_lambda": l2_lambda,
                        "print_step": 2000,
                        "results_dir": args.results_dir,
                        "experiment_name": f'{name}_hs_{hidden_size}_lr_{learning_rate}_wd_{l2_lambda}_datasz',  # Will be generated if None
                        "exp_id": exp_counter,
                    }
                configs.append(config)
                exp_counter += 1

    # List of available GPU IDs - for now just "nvidia-smi ing" it to get the list before running
    # kinda hacky but works for now
    #TODO maybe make more things command line params, its just easier for me to look at this way
    

    # Run grid search with specified GPU IDs
    perform_grid_search(configs, gpu_ids=args.available_gpu_ids)
