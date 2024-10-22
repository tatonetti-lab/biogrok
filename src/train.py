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

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

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


def train_mlp_orig(input_data, labels, val_data, val_labels, hidden_size=10, learning_rate=0.01, epochs=100, seed=42, gpu_device=None, l2_lambda=0.01, depth="shallow", print_step=100):
    # Set the seed for reproducibility
    set_seed(seed)

    # Set device to the specified GPU if available, otherwise CPU
    if gpu_device is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    input_size = input_data.shape[1]
    output_size = labels.shape[1]

    # Initialize the model, loss function, and optimizer
    if depth == "shallow":
        model = MLP(input_size, hidden_size, output_size).to(device)
    elif depth == "deep":
        model = DeepMLP(input_size, hidden_size, output_size).to(device)
    else:
        raise Exception("Error in model depth provided.")
    
    criterion = nn.MSELoss()  # Assuming we're doing regression, adjust for classification if needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)  # L2 regularization added via weight_decay

    # Convert input data and validation data to PyTorch tensors and move to device
    input_tensor = torch.FloatTensor(input_data).to(device)
    labels_tensor = torch.FloatTensor(labels).to(device)
    val_tensor = torch.FloatTensor(val_data).to(device)
    val_labels_tensor = torch.FloatTensor(val_labels).to(device)

    # Lists to store training and validation losses
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Forward pass on training data
        model.train()
        outputs = model(input_tensor)
        loss = criterion(outputs, labels_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store training loss
        training_losses.append(loss.item())

        # Calculate training accuracy
        train_mae = calculate_mae(outputs, labels_tensor)
        training_accuracies.append(train_mae)

        # Evaluate on the validation set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_labels_tensor)

        # Store validation loss
        validation_losses.append(val_loss.item())

        # Calculate validation accuracy
        val_mae = calculate_mae(val_outputs, val_labels_tensor)
        validation_accuracies.append(val_mae)

        if (epoch + 1) % print_step == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}')
        
    return {"model": model,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "training_acc": training_accuracies,
            "validation_acc": validation_accuracies}

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
                           training_mae, validation_mae, avg_test_loss, avg_test_mae,
                           )




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


def get_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_actuals.append(labels.cpu().numpy())
    return np.concatenate(all_predictions), np.concatenate(all_actuals)


def save_results_and_plots(config, model, training_losses, validation_losses,
                           training_mae, validation_mae, test_loss, test_mae,
                           ):
    """Save training results and plots to a dedicated directory for the experiment."""
    
    # Create a base results directory if it doesn't exist
    base_results_dir = config["results_dir"]
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    experiment_dir = os.path.join(base_results_dir, config['experiment_name'])

    if  os.path.exists(experiment_dir):
        i = 0
        while os.path.exists(f"{experiment_dir}_{i}"):
            i += 1
        experiment_dir = f"{experiment_dir}_{i}"

    # Create the experiment-specific directory
    os.makedirs(experiment_dir)

    # Prepare data to save in a JSON file
    save_data = {
        "config": config,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "training_mae": training_mae,
        "validation_mae": validation_mae,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "trainloader_params": config['trainloader'].dataset.get_dataset_params(),
        "valloader_params": config['valloader'].dataset.get_dataset_params(),
        "testloader_params": config['testloader'].dataset.get_dataset_params()
    }

    # Convert data to serializable format
    serializable_data = convert_to_serializable(save_data)

    # Save the results to a JSON file inside the experiment directory
    result_filename = os.path.join(experiment_dir, f"results.json")
    with open(result_filename, 'w') as f:
        json.dump(serializable_data, f, indent=4)

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


    for phase, dataloader in [('Train', config['trainloader']), ('Val', config['valloader']), ('Test', config['testloader'])]:
        
        predictions, actuals = get_predictions(model, dataloader, config['gpu_device'])

        plt.figure()
        plt.plot(predictions[:, 0], label='Predicted Prey', alpha=0.7)
        plt.plot(actuals[:, 0], label='Actual Prey', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Prey Population')
        plt.title(f'{phase}: Prey Population: Predictions vs. Actuals')
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, f"{phase}_prey_population.png"))
        plt.close()

        # Predator Predictions vs. Actuals
        plt.figure()
        plt.plot(predictions[:, 1], label='Predicted Predator', alpha=0.7)
        plt.plot(actuals[:, 1], label='Actual Predator', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Predator Population')
        plt.title(f'{phase}: Predator Population: Predictions vs. Actuals')
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, f"{phase}_predator_population.png"))
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

    gpu_processes = {gpu_id: [] for gpu_id in gpu_ids}

    for i, config in enumerate(configs):
        gpu_id = gpu_ids[i % num_gpus]
        config["gpu_device"] = gpu_id
        config["exp_id"] = i  # Assign an experiment ID
        train_mlp(config)

    #     p = multiprocessing.Process(target=train_mlp, args=(config,))
    #     gpu_processes[gpu_id].append(p)

    # for gpu_id in gpu_ids:
    #     for p in gpu_processes[gpu_id]:
    #         p.start()
    #         p.join()  # Wait for the process to finish before starting the next one on this GPU

    # # Ensure all processes have completed
    # for gpu_id in gpu_ids:
    #     for p in gpu_processes[gpu_id]:
    #         if p.is_alive():
    #             p.join()



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run grid search for MLP models.")
    parser.add_argument('-g', "--available_gpu_ids", type=int, nargs='+', default=[0, 1, 2, 3],
                        help="List of available GPU IDs (default: [0, 1, 2, 3])")
    parser.add_argument("--results_dir", type=str, default="../results/by_timepoint",
                        help="Directory to save the results (default: ../results/by_timepoint)")
    parser.add_argument("--datadir", type=str, default="../data",
                        help="Directory containing the CSV data files (default: ../data)")
    parser.add_argument('-n', "--exp_name", type=str, default=None, help="prefix for the experiment name")
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[512],
                        help="List of hidden sizes to try (default: [512])")
    parser.add_argument("--l2_lambdas", type=float, nargs='+', default=[0.0],
                        help="List of L2 regularization lambdas to try (default: [0.0])")
    parser.add_argument("--learning_rates", type=float, nargs='+', default=[0.0001],
                        help="List of learning rates to try (default: [0.0001])")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of epochs to train (default: 10000)")

    return parser.parse_args()



if __name__ == "__main__":

    """
    FOR NOW DOING MOST OF THE PARAM SETTING FOR THE GRID SEARCH MANUALLY IN HERE CUZ ITS EASIER FOR ME TO LOOK AT

    ALSO NOTE THIS IS MULTIPROCESSED SO YOUR PRINT STATEMENTS MIGHT BE INTERLEVED FOR DIFF PARAM SETS
    
    """

    args = parse_arguments()

    # get datasets
    # csv_files = os.listdir(args.datadir)
    # filenames = [os.path.join(args.datadir, f) for f in csv_files]
    batch_size = 256
    #trainloader, valloader, testloader = get_dataloaders(train_start_end=[100, 300], val_start_end=[0, 100], test_start_end=[300, 400], step_size=1.0, batch_size=batch_size)

    datafile = '../data/predator_prey_solution_12_60.csv'
    df = pd.read_csv(datafile)
    initial_conditions = np.tile(df[df["Time"]==0][["Prey", "Predator"]].to_numpy()[0], (df.shape[0], 1))
    times = df["Time"].to_numpy().reshape(-1,1)
    inputs = np.hstack((initial_conditions, times))
    outputs = df[["Prey","Predator"]].to_numpy()

    training_data = {
        'inputs': inputs[100:300,:],
        'labels': outputs[100:300,:]
    }
    print(training_data['inputs'].shape)

    validation_data = {
        'inputs': inputs[:100,:],
        'labels': outputs[:100,:]
    }
    print(validation_data['inputs'].shape)

    testing_data = {
        'inputs': inputs[300:400,:],
        'labels': outputs[300:400,:]
    }
    print(testing_data['inputs'].shape)

    # traindataset = TensorDataset(torch.tensor(training_data['inputs']), torch.tensor(training_data['labels']))
    # trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    # print(torch.tensor(training_data['inputs']).shape)

    # valdataset = TensorDataset(torch.tensor(validation_data['inputs']), torch.tensor(validation_data['labels']))
    # valloader = DataLoader(valdataset, batch_size=batch_size)

    # testdataset = TensorDataset(torch.tensor(testing_data['inputs']), torch.tensor(testing_data['labels']))
    # testloader = DataLoader(testdataset, batch_size=batch_size)

    # Define your configurations for the grid search
    configs = []
    exp_counter = 0
    # for name, model in [("mlp", MLP), ("deep_mlp", DeepMLP)]:
    for name, model in [("deep_mlp", DeepMLP),]:
        for hidden_size in args.hidden_sizes:
            for l2_lambda in args.l2_lambdas:
                for learning_rate in args.learning_rates:
                    config = {
                        # "trainloader": trainloader,
                        # "valloader": valloader,
                        # "testloader": testloader,
                        "model": model,
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "epochs": args.epochs,
                        "seed": 42,
                        "gpu_device": None,  # Will be set in perform_grid_search
                        "l2_lambda": l2_lambda,
                        "print_step": int(args.epochs/10),
                        "results_dir": args.results_dir,
                        "experiment_name": f'{args.exp_name}_hs_{hidden_size}_lr_{learning_rate}_wd_{l2_lambda}',  # Will be generated if None
                        "exp_id": exp_counter,
                    }
                    results = train_mlp_orig(training_data['inputs'], 
                        training_data['labels'], 
                        validation_data['inputs'], 
                        validation_data['labels'], 
                        hidden_size=hidden_size, 
                        learning_rate=learning_rate, 
                        epochs=args.epochs, 
                        seed=42,
                        gpu_device=args.available_gpu_ids[0],
                        l2_lambda=l2_lambda,
                        depth="deep",
                        print_step=int(args.epochs/10))
                    configs.append(config)
                    exp_counter += 1

    #TODO - why is this so slow
    #TODO - look into init conditions    

    # Run grid search with specified GPU IDs
    #perform_grid_search(configs, gpu_ids=args.available_gpu_ids)


