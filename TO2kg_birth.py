# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:56:52 2024

@author: nicol
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math

# Define parameters
simulation_time_steps = 200

# Given constants
pO2 = 21.3  # Oxygen partial pressure in kPa
alpha = 1  # Exponent for oxygen in metabolic rate equation
beta = 0.06  # Temperature coefficient
c = 1

# Function to calculate metabolic rate R_d
def R_d(M, T, pO2, alpha, beta):
    return 1.5 * (M**(2/3)) * (pO2**alpha) * np.exp(beta * T)

# Function to calculate daily intake I_d
def daily_intake(M, T, pO2, alpha, beta):
    r_d = R_d(M, T, pO2, alpha, beta)  # Call the R_d function
    return r_d  # Use the result of R_d directly for intake rate

# Function to calculate carrying capacity K
def K(NPP):
    return NPP * c

# Modified survival probability function
def survival_probability(K, daily_intake):
    return K / (daily_intake + K) if daily_intake > 0 else 0

# Define the activity rate based on metabolic rate (R_d)
def calculate_activity_rate(R_d):
    if R_d > 500:  # Threshold for high metabolic rate
        return 2.0  # High activity rate
    elif R_d > 300:  # Threshold for medium metabolic rate
        return 1.5  # Medium activity rate
    else:
        return 1.2  # Low activity rate

# # Death function with checkpoints
# def death_function(population, grid_size_lon, grid_size_lat, threshold_factor=0.5):
#     death_list = []
    
#     # Create a dictionary to store survival probabilities in each grid cell
#     grid_cells = {}
#     for individual in population:
#         grid_pos = (individual['x'], individual['y'])
#         if grid_pos not in grid_cells:
#             grid_cells[grid_pos] = []
#         grid_cells[grid_pos].append(individual)
    
#     # Iterate over each grid cell
#     for grid_pos, individuals in grid_cells.items():
#         if len(individuals) > 1:  # Only apply the rule if more than 1 individual exists in the cell
#             # Calculate the median survival probability in the grid cell
#             survival_probs = [ind['survival_probability'] for ind in individuals]
#             median_survival_prob = np.median(survival_probs)
            
#             # Compare individual's survival probability to the threshold
#             for individual in individuals:
#                 if individual['survival_probability'] < median_survival_prob * threshold_factor:
#                     death_list.append(individual)

            # Checkpoint to see if any individuals are below the death threshold
         #   print(f"Grid Cell {grid_pos}: Median Survival Prob = {median_survival_prob}, Deaths = {len(death_list)}")

    # Remove individuals marked for death
    for individual in death_list:
        population.remove(individual)

    return len(death_list)

# Read land from CSV file
land_values_lowres_df = pd.read_csv('land_cru.csv', header=None)
land_values_lowres = land_values_lowres_df.values
land_transpose = np.matrix.transpose(land_values_lowres)
land = land_transpose

# Read additional resource data from CSV file
resource_values_lowres_df = pd.read_csv('NPP.csv', header=None)
resource_values_lowres = resource_values_lowres_df.values
resource_transpose = np.matrix.transpose(resource_values_lowres)
resource_scaled = 10 * np.divide(resource_transpose, np.nanmax(resource_transpose))
resource_scaled = np.nan_to_num(resource_scaled, nan=0)
NPP = resource_scaled * 100

# Read latitude and longitude data
lat_df = pd.read_csv('lat_cru.csv', header=None)
lat = lat_df.values
lat = np.squeeze(lat)

lon_df = pd.read_csv('lon_cru.csv', header=None)
lon = lon_df.values
lon = np.squeeze(lon)

# Define the spatial grid
grid_size_lon, grid_size_lat = land.shape

# Load temperature data from CSV
temperature_data = pd.read_csv('tmp_avg.csv', header=None)

# Check the shape of the loaded data 
temperature_values = temperature_data.values
temperature_gradient = np.matrix.transpose(temperature_values)
T = np.nan_to_num(temperature_gradient, nan=999)

# Oxygen gradient
oxygen_gradient = np.linspace(21, 19, grid_size_lat)

# Initialize population with varying masses
population = []
for _ in range(10000):
    while True:
        x = np.random.randint(0, grid_size_lon)
        y = np.random.randint(0, grid_size_lat)
        if land[x, y] == 1 and NPP[x, y] > 0:  # Ensure resources exist in the land area
            break
    biomass = random.randint(1, 500)  # Assign random biomass between 1 and 500 kg
    local_T = T[x, y]
    R_d_value = R_d(biomass, local_T, pO2, alpha, beta)
    I_d = daily_intake(biomass, local_T, pO2, alpha, beta)
    local_NPP = NPP[x, y]
    K_value = K(local_NPP)
    survival_probability_value = survival_probability(K_value, I_d)

    # Checkpoint for each individual
  #  print(f"Initial Individual (x={x}, y={y}): Biomass = {biomass}, Survival Probability = {survival_probability_value}")

    individual = {
        'x': x,
        'y': y,
        'biomass': biomass,
        'survival_probability': survival_probability_value,
        'R_d': R_d_value,
        'age': 0  # Initialize age to 0
    }
    population.append(individual)

# Simulation
total_population_size = []
num_individuals_over_time = []
total_population_biomass = []

# Movement logic
for t in range(simulation_time_steps):
    new_population = []
    num_births = 0
    num_deaths = 0
    num_moves = 0

    age_survival_decay_rate = 0.01  # Amount to reduce survival probability per time step based on age

    for individual in population:
        # Ensure each individual has an 'age' attribute
        if 'age' not in individual:
            individual['age'] = 0  # Initialize age if not present

        # Increment age
        individual['age'] += 1

        # Update metabolic rate and activity rate
        individual['R_d'] = R_d(individual['biomass'], T[individual['x'], individual['y']], pO2, alpha, beta)
        activity_rate = calculate_activity_rate(individual['R_d'])

        # Recalculate movement probability based on survival_probability and activity rate
        if individual['survival_probability'] > 0.75:
            movement_probability = 0.1 * activity_rate  # Lower movement probability for high survival, scaled by activity rate
        elif individual['survival_probability'] > 0.5:
            movement_probability = 0.3 * activity_rate  # Medium movement probability
        else:
            movement_probability = 0.5 * activity_rate  # Higher movement probability for low survival

        # Randomly decide whether the individual moves based on movement_probability
        if random.random() < movement_probability:
            movement_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            dx, dy = movement_options[random.randint(0, 3)]
            
            # Test the new position
            test_move_x = individual['x'] + dx
            test_move_y = individual['y'] + dy
            
            if 0 <= test_move_x < grid_size_lon and 0 <= test_move_y < grid_size_lat:
                if land[test_move_x, test_move_y] == 1 and resource_scaled[test_move_x, test_move_y] > 0:
                    individual['x'] = test_move_x
                    individual['y'] = test_move_y
                    num_moves += 1

                    # Recalculate survival probability after movement
                    individual['R_d'] = R_d(individual['biomass'], T[individual['x'], individual['y']], pO2, alpha, beta)
                    K_value = K(NPP[individual['x'], individual['y']])
                    I_d = daily_intake(individual['biomass'], T[individual['x'], individual['y']], pO2, alpha, beta)
                    individual['survival_probability'] = survival_probability(K_value, I_d)

        # Apply age-based decay to survival probability
        individual['survival_probability'] -= age_survival_decay_rate * individual['age']
        individual['survival_probability'] = max(0, individual['survival_probability'])  # Ensure it doesn't go below 0

    # Calculate population density in terms of biomass
    population_density = np.zeros((grid_size_lon, grid_size_lat))
    for individual in population:
        if 0 <= individual['x'] < grid_size_lon and 0 <= individual['y'] < grid_size_lat:
            population_density[individual['x'], individual['y']] += individual['biomass']

    # Adjust resources based on population density
    adjusted_resources = np.copy(resource_scaled)

    # Apply the death function
    num_deaths = death_function(population, grid_size_lon, grid_size_lat)

    # Checkpoint for total deaths at each time step
  #  print(f"Time Step {t + 1}: Total Deaths = {num_deaths}")

  

    # Function to perform reproduction based on proximity, survival probability, and mass compatibility
    def birth(population):
        num_new_individuals = 0
        new_individuals = []
    
        for individual in population:
            # Calculate spatial distance-based mating probability
            spatial_distances = []
            valid_mating_partners = []  # List of individuals that satisfy mass criteria
    
            for other_individual in population:
                if other_individual != individual:
                    # Check mass compatibility (within 50% of each other's mass)
                    if 0.5 * individual['biomass'] <= other_individual['biomass'] <= 1.5 * individual['biomass']:
                        # If mass is within range, calculate distance
                        distance = math.sqrt((individual['x'] - other_individual['x']) ** 2 + (individual['y'] - other_individual['y']) ** 2)
                        spatial_distances.append(distance)
                        valid_mating_partners.append(other_individual)
            
            # Check if there are any valid mating partners based on mass
            if valid_mating_partners:
                # Normalize spatial distances to get mating probabilities
                sum_spatial_distances = sum(spatial_distances)
                if sum_spatial_distances > 0:
                    normalized_spatial_distances = [1 - (dist / sum_spatial_distances) for dist in spatial_distances]  # Closer individuals have higher probability
                else:
                    normalized_spatial_distances = [1] * len(spatial_distances)  # If no movement, equal probability
            else:
                # No valid partners found, skip reproduction for this individual
                continue
            
            # Calculate the reproduction probability using spatial proximity and individual survival probability
            reproduction_probability = 0.8 * individual['survival_probability']  # Base probability adjusted by survival
    
            # Check if the individual reproduces (randomly based on reproduction probability)
            if random.random() < reproduction_probability:
                # Choose a valid partner randomly (weighted by proximity if desired)
                chosen_partner = random.choices(valid_mating_partners, weights=normalized_spatial_distances, k=1)[0]
                
                # Combine parents' locations (random point between them)
                offspring_x = (individual['x'] + chosen_partner['x']) / 2 + random.uniform(-5, 5)  # Add some randomness
                offspring_y = (individual['y'] + chosen_partner['y']) / 2 + random.uniform(-5, 5)
    
                # Create new individual with random biomass and place it near the parent's location
                offspring_biomass = random.randint(1, 500)  # Random mass for offspring
                offspring = {
                    'x': offspring_x,  # Offspring placed at parent's location
                    'y': offspring_y, 
                    'biomass': offspring_biomass,
                    'age': 0  # Newborn starts with age 0
                }
                # Calculate offspring's initial survival probability
                K_value = K(NPP[offspring['x'], offspring['y']])
                I_d = daily_intake(offspring['biomass'], T[offspring['x'], offspring['y']], pO2, alpha, beta)
                offspring['survival_probability'] = survival_probability(K_value, I_d)
    
                # Add new individual to the list
                new_individuals.append(offspring)
                num_new_individuals += 1
        
        # Add all new offspring to the population
        population.extend(new_individuals)

        return num_new_individuals


    population.extend(new_population)
    
    

    # Corrected print statement
    print(f"Time Step {t + 1}: Births = {num_births}, Deaths = {num_deaths}, Moves = {num_moves}, Biomass of Population = {sum(ind['biomass'] for ind in population)} kg")

    # Store total biomass for plotting later
    total_biomass = sum(ind['biomass'] for ind in population)
    num_individuals_over_time.append(total_biomass)
    
    # Track population size and biomass over time
    total_population_size.append(len(population))  # Number of individuals
    total_population_biomass.append(sum(ind['biomass'] for ind in population))  # Total biomass


    # Plotting code here (as before)
    plt.figure(figsize=(15, 7))

    # Plot population biomass density
    plt.subplot(1, 3, 1)
    plt.imshow(population_density, aspect='auto', cmap='viridis', extent=[0, grid_size_lon, 0, grid_size_lat])
    plt.colorbar(label='Biomass Density (kg)')
    plt.title('Biomass Density')

    # Plot adjusted resources
    plt.subplot(1, 3, 2)
    plt.imshow(adjusted_resources, aspect='auto', cmap='terrain', extent=[0, grid_size_lon, 0, grid_size_lat])
    plt.colorbar(label='Resources')
    plt.title('Adjusted Resources')

    # Plot land
    plt.subplot(1, 3, 3)
    plt.imshow(land, aspect='auto', cmap='viridis', extent=[0, grid_size_lon, 0, grid_size_lat])
    plt.colorbar(label='Land')
    plt.title('Land')

    plt.show()

# Final population against time plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, simulation_time_steps + 1), total_population_size, label='Population Size', color='blue', marker='o')
plt.plot(range(1, simulation_time_steps + 1), total_population_biomass, label='Total Biomass (kg)', color='green', marker='x')

plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.title('Population Size and Biomass Over Time')
plt.legend(loc='best')
plt.grid(True)
plt.show()
