

import numpy as np
import pandas as pd


def build_ppt_datasets(csv_files, n_datapoints, train_val_test_split=(0.8, 0.1, 0.1)):
    """
    Builds a dataset from a list of csv file paths.
    inputs are the initial conditions and time
    outputs are the prey and predator counts at given time point
    """
    inputs_list = list()
    outputs_list = list()

    for file in csv_files:
        df = pd.read_csv(file)
        initial_conditions = np.tile(df[df["Time"]==0][["Prey", "Predator"]].to_numpy()[0], (df.shape[0], 1))
        times = df["Time"].to_numpy().reshape(-1,1)
        inputs_list.append( np.hstack((initial_conditions, times)) )
        outputs_list.append( df[["Prey","Predator"]].to_numpy() )

    inputs = np.vstack(inputs_list)
    outputs = np.vstack(outputs_list)

    # Shuffle the data
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    if n_datapoints != 'all':
        indices = indices[:int(n_datapoints)]
    inputs = inputs[indices]
    outputs = outputs[indices]

    # Calculate split indices
    total_samples = inputs.shape[0]
    train_split, val_split, _ = train_val_test_split
    train_idx = int(total_samples * train_split)
    val_idx = train_idx + int(total_samples * val_split)

    # Split the data
    train_inputs, val_inputs, test_inputs = np.split(inputs, [train_idx, val_idx])
    train_outputs, val_outputs, test_outputs = np.split(outputs, [train_idx, val_idx])

    print("Datasets generated with sizes:")
    print(f"Train: {train_inputs.shape}, Val: {val_inputs.shape}, Test: {test_inputs.shape}")

    return {
        "train": {'inputs': train_inputs, 'labels': train_outputs},
        "val": {'inputs': val_inputs, 'labels': val_outputs},
        "test": {'inputs': test_inputs, 'labels': test_outputs}
    }



   