import pandas as pd
import os
# Metrics
import math
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
# Data Transformation
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


ROOT_DIR = Path(str(os.getcwd()))
SAVE_DIR =  ROOT_DIR / 'output'
PLOT_SAVE_DIR = SAVE_DIR / 'plots'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(PLOT_SAVE_DIR):
    os.makedirs(PLOT_SAVE_DIR)


def plot_numberic_column_dist(data, PLOT_SAVE_DIR):
    # Select numerical columns from the DataFrame
    numerics = data.select_dtypes(include='number')

    # Calculate the number of plots, rows, and columns for subplots
    num_plots = len(numerics.columns)
    num_columns = 3
    num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)
    # Set the figure size based on the number of rows
    plt.figure(figsize=(10, 4 * num_rows))

    # Iterate over each numerical column and create a histogram subplot
    for i, col in enumerate(numerics, 1):
        plt.subplot(num_rows, num_columns, i)  # Create subplot
        mean_values = numerics[col].mean()
        median = numerics[col].median()

        sns.histplot(numerics[col], kde=True, color='#638889')  # Plot histogram using seaborn
        plt.axvline(x=mean_values, color='#F28585', linestyle='--', label='Mean')
        plt.axvline(x=median, color='#747264', linestyle='--', label='Median')
        plt.grid(True, alpha=0.8)  # Add grid lines to the plot
        plt.title(f'{col} Distribution')  # Set title for the subplot
        plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(PLOT_SAVE_DIR / 'numberic_column_dist.png')
    plt.show()  # Display the plots
    return

def plot_categorical_column_dist(data, PLOT_SAVE_DIR):
    # Select categorical columns from the DataFrame
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Calculate the number of plots, rows, and columns for subplots
    num_plots = len(categorical_cols)
    num_columns = 3
    num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)

    # Set the figure size based on the number of rows
    plt.figure(figsize=(10, 4 * num_rows))

    # Iterate over each categorical column and create a histogram subplot
    for i, col in enumerate(data[categorical_cols], 1):
        mode = data[col].mode()[0]    
        plt.subplot(num_rows, num_columns, i)  # Create subplot
        sns.histplot(data[col], kde=True, color='#638889')  # Plot histogram using seaborn

        plt.axvline(x=mode, color='#F28585', linestyle='--', label='Mode')

        plt.xticks(rotation=90, fontsize=7)  # Rotate x-axis labels for better readability
        plt.title(f'{col} Distribution')  # Set title for the subplot

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(PLOT_SAVE_DIR / 'categorical_column_dist.png')
    plt.show()  # Display the plots
    return


def get_z_score_plot_numerical_columns(data, PLOT_SAVE_DIR):
    numerics = data.select_dtypes(include=np.number)

    # Calculate the number of plots, rows, and columns for subplots
    num_plots = len(numerics.columns)
    num_columns = 3
    num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)

    # Set the figure size based on the number of rows
    plt.figure(figsize=(10, 4 * num_rows))

    for i, col in enumerate(numerics, 1):
        plt.subplot(num_rows, num_columns, i)  
        z_scores = (numerics[col] - numerics[col].mean()) / numerics[col].std()

        threshold = 3

        plt.scatter(np.arange(len(z_scores)), z_scores, color='#638889', alpha=0.5)
        plt.axhline(y=threshold, color='#F28585', linestyle='--', label='Threshold')
        plt.axhline(y=-threshold, color='#F28585', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Z-score')
        plt.title(f'Z-score Plot for {col}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_DIR / 'z_score_plot_numerical_columns.png')
    plt.show()
    return


def get_corelation_plot(data, PLOT_SAVE_DIR):
    # Set the figure size
    plt.figure(figsize=(6, 4))

    # Create a heatmap of the correlation matrix for numerical columns in the dataframe
    sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, 
                cmap=['#638889', '#678788', '#6c8788', '#718788', '#768788',
                    '#7b8788', '#808788', '#858788', '#8a8787', '#8f8787',
                    '#948687', '#998687', '#9e8687', '#a38687', '#a88687',
                    '#ac8686', '#b18686', '#b68686', '#bb8686', '#c08686',
                    '#c58586', '#ca8586', '#cf8585', '#d48585', '#d98585',
                    '#de8585', '#e38585', '#e88585', '#ed8585', '#f28585'], annot_kws={"fontsize":8})

    # Show the plot
    plt.savefig(PLOT_SAVE_DIR / 'corelation_plot.png')
    plt.show()
    return


def get_scatter_plot_cost_of_ride_expected_ride_duration(data, PLOT_SAVE_DIR):
    # Create a scatter plot with linear regression lines using seaborn's lmplot
    sns.lmplot(data=data, y='Historical_Cost_of_Ride', x='Expected_Ride_Duration', hue='Vehicle_Type', 
            palette=['#638889', '#f28585'])

    # Show the plot
    plt.savefig(PLOT_SAVE_DIR / 'scatter_plot_Historical_Cost_of_RideExpected_Ride_Duration.png')
    plt.show()
    return


def main(data_csv):
    data = data_csv.copy()
    plot_numberic_column_dist(data, PLOT_SAVE_DIR)
    plot_categorical_column_dist(data, PLOT_SAVE_DIR)
    get_z_score_plot_numerical_columns(data, PLOT_SAVE_DIR)
    get_corelation_plot(data, PLOT_SAVE_DIR)
    get_scatter_plot_cost_of_ride_expected_ride_duration(data, PLOT_SAVE_DIR)

    return


if __name__ == '__main__':
    data_file = Path(r'D:\ml_projects\dynamicPricing\dynamic_pricing.csv')
    data = pd.read_csv(data_file)
    main(data)