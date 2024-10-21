import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
from torch.utils.data import Dataset, DataLoader

# Define the Lotka-Volterra equations
def lotka_volterra(y, t, alpha, beta, delta, gamma):
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

class PredatorPreyDataset(Dataset):
    def __init__(self, 
                 alpha=1.0, beta=0.1, delta=0.075, gamma=1.5, 
                 prey_ic=50, pred_ic=30, 
                 start_step=0, end_time=400, step_size=1.0):
        """
        Initializes the dataset for Lotka-Volterra simulations.
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.prey_ic = prey_ic
        self.pred_ic = pred_ic
        self.end_time = end_time
        self.start_step = start_step
        self.step_size = step_size

        # Generate the full dataset once during initialization
        self.time, self.prey_population, self.predator_population = self._generate_data()


    def _generate_data(self):
        """
        Solves the Lotka-Volterra ODE and returns the solution.
        """
        initial_conditions = [self.prey_ic, self.pred_ic]
        time = np.linspace(0, self.end_time, int(self.end_time/self.step_size))

        # Solve the ODE
        solution = odeint(lotka_volterra, initial_conditions, time, 
                          args=(self.alpha, self.beta, self.delta, self.gamma))
        prey_population, predator_population = solution.T

        return time[self.start_step:self.end_time], prey_population[self.start_step:self.end_time], predator_population[self.start_step:self.end_time]

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        
        x = torch.tensor([self.time[idx]], dtype=torch.float32)
        y = torch.tensor([self.prey_population[idx], self.predator_population[idx]], dtype=torch.float32)
        return x, y

    def plot_solution(self):
        """
        Plots the predator-prey solution.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, 
                 self.prey_population, 
                 label='Prey Population', color='blue')
        plt.plot(self.time, 
                 self.predator_population, 
                 label='Predator Population', color='red')
        plt.title('Predator-Prey Dynamics (Lotka-Volterra Model)')
        plt.xlabel('Time')
        plt.ylabel('Population')
        # Calculate 4 evenly spaced tick locations
        tick_locations = np.linspace(self.time[0], self.time[-1], 4)
        
        # Calculate corresponding tick labels
        tick_labels = np.linspace(self.start_step, self.start_step + len(self.time) * self.step_size, 4, dtype=int)
        
        plt.xticks(tick_locations, tick_labels)
        plt.legend()
        plt.grid()
        plt.show()    

    #serialize and save dataloader params
    def get_dataset_params(self):
        """
        Serializes and returns the dataloader parameters as a dictionary.
        """
        params = {
            'prey_ic': self.prey_ic,
            'pred_ic': self.pred_ic,
            'alpha': self.alpha,
            'beta': self.beta,
            'delta': self.delta,
            'gamma': self.gamma,
            'start_step': self.start_step,
            'end_time': self.end_time,
            'step_size': self.step_size
        }
        return params

def get_dataloaders(train_start_end=[100, 300], val_start_end=[0, 100], test_start_end=[300, 400], step_size=1.0, batch_size=32):
    trainset = PredatorPreyDataset(start_step=train_start_end[0], end_time=train_start_end[1], step_size=step_size)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    
    valset = PredatorPreyDataset(start_step=val_start_end[0], end_time=val_start_end[1], step_size=step_size)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    testset = PredatorPreyDataset(start_step=test_start_end[0], end_time=test_start_end[1], step_size=step_size)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, testloader