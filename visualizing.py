import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
from datetime import datetime


def calculate_pvalues_matrix(series):
    """
    Calculate the p-value matrix for the series.
    Returns a dataframe with the matrix of p-values.
    """
    indices = series.index
    matrix = np.zeros((len(indices), len(indices)))

    for i, idx1 in enumerate(indices):
        for j, idx2 in enumerate(indices):
            if idx1 == idx2:
                matrix[i, j] = 1.0  # p-value is 1 for identical series
            else:
                _, p_val = ttest_ind(series[idx1], series[idx2], nan_policy='omit')  # Using nan_policy to handle NaNs
                matrix[i, j] = p_val

    pvalues_df = pd.DataFrame(matrix, columns=indices, index=indices)

    # Map the numeric index/columns to the desired "Sx" format
    mapper = {idx: f"S{i+1}" for i, idx in enumerate(indices)}
    pvalues_df = pvalues_df.rename(index=mapper, columns=mapper)

    return pvalues_df


def visualize_data(experimental_dataframe='exp_result.csv', threshold_list=None, plot_kind='violin',
                   colormap=None, difference=False, top_n=None):

    current_time = datetime.now()
    folder_name = 'main_figures/' + current_time.strftime('%Y_%m_%d_%H') + '_figure'
 
    if not os.path.exists('main_figures'):
        os.makedirs('main_figures')

    # Check if the folder doesn't already exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # Input validation
    if top_n is not None and difference:
        raise ValueError("Only one of top_n and difference can be provided. At least one of them should be None.")

    threshold_list = threshold_list or []

    # Default colormap
    colormap = colormap or {
        0: '#D631FF', 1: '#FF8000', 2: '#FF0000',
        3: '#3CFF3C', 4: '#31FFFF', 5: '#0000FF'
    }

    df1 = pd.read_csv(experimental_dataframe)
    new_labels = ["S" + str(i) for i in range(1, len(df1['sequence']) + 1)]

    def drop_sequence_column(df):
        return df.drop(columns=['sequence']) if 'sequence' in df.columns else df

    n = len(threshold_list)
    n_rows = (n // 2) + (n % 2)
    if not plot_kind=='heatmap':

        fig, axs = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5 + 2))

    all_pvalue_matrices = []

    for idx, imm in enumerate(threshold_list):
        df2 = pd.read_csv(imm)
        numeric_common_columns = df1.select_dtypes(include='number').columns.intersection(df2.columns)
        numeric_difference = drop_sequence_column(df2[numeric_common_columns]) - drop_sequence_column(df1[numeric_common_columns])

        if difference:
            plot_data_df2 = numeric_difference.apply(lambda row: row.values, axis=1)
        elif top_n:
            abs_diff = numeric_difference.abs()
            top_n_cols_per_row = abs_diff.apply(lambda row: row.nsmallest(top_n).index, axis=1)
            top_n_diff = numeric_difference.apply(lambda row: row[top_n_cols_per_row[row.name]], axis=1)
            non_nan_cols = top_n_diff.apply(lambda row: row.dropna().index, axis=1)
            plot_data_df2 = df2.apply(lambda row: row[non_nan_cols[row.name]].values, axis=1)
        else:
            plot_data_df2 = drop_sequence_column(df2[numeric_common_columns]).apply(lambda row: row.values, axis=1)

        if not plot_kind=='heatmap':
            ax = axs[idx // 2, idx % 2]

        if plot_kind == 'boxplot':
            boxplots = ax.boxplot(plot_data_df2.values, patch_artist=True)
            df_tmp = pd.DataFrame(plot_data_df2.tolist()).T
            df_tmp.columns = [f"S{i+1}" for i in range(df_tmp.shape[1])]
            df_tmp.to_csv(f'{plot_kind}_top_n_{top_n}_difference_{difference}_{imm}', index=False)
            for i, patch in enumerate(boxplots['boxes']):
                patch_color = colormap.get(i % len(colormap), '#000000')  # Default to black if color not found
                patch.set_facecolor(patch_color)
            for median in boxplots['medians']:
                median.set_color('darkblue')
                median.set_linewidth(1)
            ax.set_xticks(range(1, len(new_labels) + 1))
            ax.set_xticklabels(new_labels)
        elif plot_kind == 'violin':
            boxplots = ax.boxplot(plot_data_df2.values, patch_artist=True)
            df_tmp = pd.DataFrame(plot_data_df2.tolist()).T
            df_tmp.columns = [f"S{i+1}" for i in range(df_tmp.shape[1])]
            df_tmp.to_csv(f'{plot_kind}_top_n_{top_n}_difference_{difference}_{imm}', index=False)
            sns.violinplot(data=plot_data_df2, palette=colormap, ax=ax)
            ax.set_xticks(range(len(new_labels) ))
            ax.set_xticklabels(new_labels)
        elif plot_kind == 'heatmap':
            pvalue_matrix = calculate_pvalues_matrix(plot_data_df2)
            all_pvalue_matrices.append(pvalue_matrix)
        if not plot_kind=='heatmap':

            if difference:
                ax.set_ylabel('ΔF/F difference', fontsize=16)
                ax.set_ylim(-1.3, 0.6)
            else:
                ax.set_ylabel('ΔF/F', fontsize=16)
                ax.set_ylim(0.2, 1.6)
            if top_n:
                ax.set_title(f'{imm[:-4]} - top {top_n} wavelengths')
            else:
                ax.set_title(f'{imm[:-4]} - {len(numeric_common_columns)} wavelengths')





    if plot_kind == 'heatmap':
        # Set the subplot dimensions
        subplot_height = 4  # Height of each subplot
        subplot_width = 6  # Width of each subplot to make cells wider

        # Set the font for the title and ticks without changing the global font size
        title_fontsize = 12  # Adjust this value as needed for title
        ticks_fontsize = 11  # Adjust this value as needed for ticks
        plt.rcParams['font.family'] = 'Arial'

        # Create a figure with subplots with adjusted spacing to allow for wider cells
        fig_heatmap, axs_heatmap = plt.subplots(n_rows, 2, figsize=(subplot_width * 2, subplot_height * n_rows),
                                                dpi=300, gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
        for idx, matrix in enumerate(all_pvalue_matrices):
            ax = axs_heatmap[idx // 2, idx % 2]

            # Draw the heatmap with increased font size for annotations
            sns.heatmap(matrix, annot=True, cmap="YlGnBu", cbar=False, ax=ax, fmt=".2f",
                        annot_kws={'size': 13})  # Increase annotation font size here

            if top_n:
                ax.set_title(f'{threshold_list[idx][:-4]} - top {top_n} wavelengths', fontsize=title_fontsize)
            else:
                title_list = threshold_list[idx][:-4].split('_')
                ax.set_title(f'imm = {title_list[1]}, tol = {title_list[3]}', fontsize=title_fontsize)

            # Set the fontsize for ticks
            ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

        # Remove the last axis if there's an odd number of subplots
        if n % 2 != 0:
            fig_heatmap.delaxes(axs_heatmap[n_rows - 1, 1])

        # Adjust the layout to fit the figure area nicely
        plt.tight_layout(pad=0)

        plt.show()



    plt.savefig(f'{folder_name}/{plot_kind}_top_n_{top_n}_difference_{difference}.png',dpi=300)
