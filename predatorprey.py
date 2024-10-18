import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the Lotka-Volterra equations
def lotka_volterra(y, t, alpha, beta, delta, gamma):
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

# Parameters for the model
alpha = 1.0   # Prey birth rate
beta = 0.1    # Predation rate
delta = 0.075 # Reproduction rate of predators per prey eaten
gamma = 1.5   # Predator death rate

for _ in range(10):
    # Initial conditions: [prey population, predator population]
    prey_ic = random.randint(5, 100)
    pred_ic = random.randint(5, 100)
    initial_conditions = [prey_ic, pred_ic]

    # Time points where the solution is computed
    time = np.linspace(0, 200, 1000)

    # Solve the ODE
    solution = odeint(lotka_volterra, initial_conditions, time, args=(alpha, beta, delta, gamma))
    prey_population, predator_population = solution.T

    # Save the solution matrix to a file
    solution_matrix = np.column_stack((time, prey_population, predator_population))
    np.savetxt(f'data/predator_prey_solution_{prey_ic}_{pred_ic}.csv', solution_matrix, delimiter=',', header='Time,Prey,Predator', comments='')

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time, prey_population, label='Prey Population', color='blue')
    plt.plot(time, predator_population, label='Predator Population', color='red')
    plt.title('Predator-Prey Dynamics (Lotka-Volterra Model)')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f'results/predator_prey_solution_{prey_ic}_{pred_ic}.png')