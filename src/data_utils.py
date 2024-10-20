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
    train_split, val_split, test_split = train_val_test_split
    if n_datapoints == 'all':
        n_datapoints = inputs.shape[0]
    else:
        n_datapoints = int(n_datapoints)

    test_len = int(n_datapoints * test_split)
    train_val_len = n_datapoints - test_len
    # Shuffle the data, but save some ordered test data
    # TODO maybe sort out a better way to isolate test data 
    test_inputs = inputs[-test_len:]
    test_outputs = outputs[-test_len:]

    indices = np.arange(train_val_len)
    np.random.shuffle(indices)
    inputs = inputs[indices]
    outputs = outputs[indices]
    
    train_idx = int(n_datapoints * train_split)
    # Split the data
    train_inputs = inputs[:train_idx]
    val_inputs = inputs[train_idx:]
    train_outputs = outputs[:train_idx]
    val_outputs = outputs[train_idx:]
    

    print("Datasets generated with sizes:")
    print(f"Train: {train_inputs.shape}, Val: {val_inputs.shape}, Test: {test_inputs.shape}")

    return {
        "train": {'inputs': train_inputs, 'labels': train_outputs},
        "val": {'inputs': val_inputs, 'labels': val_outputs},
        "test": {'inputs': test_inputs, 'labels': test_outputs}
    }