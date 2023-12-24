import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def scatter_plot(threshold_list):
    # Set the global font size
    plt.rcParams.update({'font.size': 14})


    current_time = datetime.now()

    # Format the date and time to the desired format
    folder_name = 'scatter_figures/' + current_time.strftime('%Y_%m_%d_%H') + '_scatter'

    # Check if the folder doesn't already exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists('scatter_figures'):
            os.makedirs('scatter_figures')



    # Read the csv files
    for threshold_index in threshold_list:
        df1 = pd.read_csv('exp_result.csv')
        df2 = pd.read_csv(f'{threshold_index}.csv')

        # Set 'sequence' as the index of the dataframes
        df1.set_index('sequence', inplace=True)
        df2.set_index('sequence', inplace=True)

        # Get the common columns
        common_cols = df1.columns.intersection(df2.columns)

        # Create a figure with 3 rows and 2 columns of subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 20), dpi=150)

        # Flatten the array of axes
        axs = axs.flatten()

        # Prepare legend handles
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=10, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=10, markeredgecolor='black'),
        ]
        labels = [f"Experimental ({len(df1.columns)} wavelengths)", f"Predicted Result ({len(df2.columns)} wavelengths)"]

        # Iterate over the sequences and the axes
        for ax, sequence in zip(axs, df1.index):
            # Get the data for this sequence
            sequence_data_df1 = df1.loc[sequence]
            sequence_data_df2 = df2.loc[sequence, common_cols] if sequence in df2.index else pd.Series()

            # Plot the data
            ax.scatter(sequence_data_df1.index, sequence_data_df1, color='darkorange', edgecolors='black')
            ax.scatter(sequence_data_df2.index, sequence_data_df2, color='darkgreen', edgecolors='black')
            ax.set_title(sequence)

            # Set x label, y label, and x ticks
            ax.set_xlabel('wavelength (nm)')
            ax.set_ylabel('ΔF/F')
            ax.set_xticks(sequence_data_df1.index[::100])
            ax.set_xticklabels(sequence_data_df1.index[::100], rotation=45, ha='right')

        # Handle the remaining axes
        for ax in axs[len(df1.index):]:
            ax.axis('off')

        # Layout adjustment
        plt.tight_layout()

        # Show the legend at the bottom center of the figure
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

        plt.savefig(f'{folder_name}/{threshold_index}.png', bbox_inches='tight', dpi=150)
    # Show the plot
    # plt.show()

