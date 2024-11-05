"""
## Feature Engineering

1. **Dynamic Pricing Calculation**

This section of code calculates the dynamic pricing adjustments for rides based on demand and supply factors. Here's a breakdown of the process:

2. **Demand Multiplier Calculation:**

high_demand_percentile and low_demand_percentile are defined to set the percentiles for high and low demand, respectively. <br>
The demand_multiplier is calculated based on the percentile for high and low demand. For each ride, it compares the number of riders (Number_of_Riders) with the respective percentiles. If the number of riders is above the high demand percentile, it divides the number of riders by the high demand percentile value. If it's below the low demand percentile, it divides by the low demand percentile value. <br>

3. **Supply Multiplier Calculation:**

Similar to demand, high_supply_percentile and low_supply_percentile are defined for setting the percentiles for high and low supply, respectively. <br>
The supply_multiplier is calculated based on the percentile for high and low supply. For each ride, it compares the number of drivers (Number_of_Drivers) with the respective percentiles. If the number of drivers is above the low supply percentile, it divides the high supply percentile value by the number of drivers. If it's below the low supply percentile, it divides the low supply percentile value by the number of drivers. <br>

4. **Price Adjustment Factors:**

demand_threshold_high, demand_threshold_low, supply_threshold_high, and supply_threshold_low are defined to set the thresholds for high and low demand/supply.  <br>

5. **Adjusted Ride Cost Calculation:**

The adjusted_ride_cost is calculated based on the historical cost of the ride (Historical_Cost_of_Ride) and the dynamic pricing adjustments determined by the demand and supply multipliers. It takes the maximum of demand multiplier and demand threshold low, and the maximum of supply multiplier and supply threshold high, then multiplies these values with the historical cost of the ride.
Overall, this process adjusts the ride cost dynamically based on demand and supply conditions to optimize pricing and maximize revenue.


"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
# Metrics
import math
from pathlib import Path
import numpy as np



ROOT_DIR = Path(str(os.getcwd()))
SAVE_DIR =  ROOT_DIR / 'output'
PLOT_SAVE_DIR = SAVE_DIR / 'plots'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(PLOT_SAVE_DIR):
    os.makedirs(PLOT_SAVE_DIR)


def get_demand_thresholds_pot_plots(data, PLOT_SAVE_DIR):
    df = data.copy()
    # Calculate demand_multiplier based on percentile for high and low demand
    high_demand_percentile = 75
    low_demand_percentile = 25

    df['demand_multiplier'] = np.where(df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                    df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                    df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand_percentile))

    # Calculate supply_multiplier based on percentile for high and low supply
    high_supply_percentile = 75
    low_supply_percentile = 25

    df['supply_multiplier'] = np.where(df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], 
                                                                            low_supply_percentile),
                                    np.percentile(df['Number_of_Drivers'], high_supply_percentile)
                                    / df['Number_of_Drivers'], np.percentile(df['Number_of_Drivers'], 
                                                                                low_supply_percentile) 
                                    / df['Number_of_Drivers'])

    # Define price adjustment factors for high and low demand/supply
    demand_threshold_high = 1.2  # Higher demand threshold
    demand_threshold_low = 0.8  # Lower demand threshold
    supply_threshold_high = 0.8  # Higher supply threshold
    supply_threshold_low = 1.2  # Lower supply threshold

    # Calculate adjusted_ride_cost for dynamic pricing
    df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (
        np.maximum(df['demand_multiplier'], demand_threshold_low) *
        np.maximum(df['supply_multiplier'], supply_threshold_high)
    )
    # Calculate the profit percentage for each ride
    df['profit_percentage'] = ((df['adjusted_ride_cost'] - df['Historical_Cost_of_Ride']) 
                            / df['Historical_Cost_of_Ride']) * 100
    # Identify profitable rides where profit percentage is positive
    profitable_rides = df[df['profit_percentage'] > 0]

    # Identify loss rides where profit percentage is negative
    loss_rides = df[df['profit_percentage'] < 0]

    # Calculate the count of profitable and loss rides
    profitable_count = len(profitable_rides)
    loss_count = len(loss_rides)

    # Create a donut chart to show the distribution of profitable and loss rides
    labels = ['Profitable Rides', 'Loss Rides']
    values = [profitable_count, loss_count]

    plt.figure(figsize=(4, 4))

    # Create a pie chart
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors = ['#638889', '#f28585'], labeldistance = 1.1,
                    pctdistance = 0.85, normalize=True
    )

    # Draw a circle in the center to create a ring
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title('Profitability of Rides (Dynamic Pricing vs. Historical Pricing)')

    plt.savefig(PLOT_SAVE_DIR / 'Dynamic_Pricing_vs_Historical_Pricing.png')
    plt.show()
    get_histogram_plot_with_KDE(df)
    get_log_transformed(df)
    return


def get_histogram_plot_with_KDE(df):
    # Calculate the skewness of the target variable 'TARGET' and round the result to 2 decimal places
    print("Skewness: ", round(df['adjusted_ride_cost'].skew(), 2))
    # Create a new figure with a specified size (8 inches width, 6 inches height)
    plt.figure(figsize=(6, 4))
    # Calculate the mean and median of the 'adjusted_ride_cost' column
    mean_values = df['adjusted_ride_cost'].mean()
    median = df['adjusted_ride_cost'].median()

    # Add vertical lines for mean and median to the plot
    plt.axvline(x=mean_values, color='#F28585', linestyle='--', label='Mean')
    plt.axvline(x=median, color='#747264', linestyle='--', label='Median')

    # Create a histogram plot with KDE (Kernel Density Estimation)
    sns.histplot(df['adjusted_ride_cost'], kde=True, color='#638889')

    # Add grid lines to the plot
    plt.grid(True)

    # Display the plot
    plt.show()
    return 

def get_log_transformed(df):
    # Apply the natural logarithm transformation plus 1 to the target variable 'TARGET'
    df['adjusted_ride_cost'] = np.log1p(df['adjusted_ride_cost'])
    # Create a new figure with a specified size (8 inches width, 6 inches height)
    plt.figure(figsize=(5, 4))

    # Calculate the mean and median of the 'adjusted_ride_cost' column
    mean_value = df['adjusted_ride_cost'].mean()
    median = df['adjusted_ride_cost'].median()

    # Plot vertical lines indicating mean and median on the histogram plot
    plt.axvline(x=mean_value, color='#F28585', linestyle='--', label='Mean')
    plt.axvline(x=median, color='#747264', linestyle='--', label='Median')

    # Create a histogram plot with KDE (Kernel Density Estimation)
    sns.histplot(df['adjusted_ride_cost'], kde=True, color='#638889')

    # Add grid lines to the plot
    plt.grid(True)

    # Display the plot
    plt.show()

    return