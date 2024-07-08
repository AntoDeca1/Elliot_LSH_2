import os
# NEW
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from pathlib import Path


def plot_and_save_results(trials, config, print_csv=True):
    # DataFrame to store all the data
    cutoff = config["experiment"]["evaluation"]["cutoffs"]
    # Determine the number of subplots needed

    # Create a results directory if it doesn't exist
    results_dir = 'results_lsh/experiments'
    # results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)

    results_subdir = config["experiment"]["dataset"]

    final_path = os.path.join(results_dir, results_subdir)
    os.makedirs(final_path, exist_ok=True)

    for model_name, experiments in trials.items():
        metrics_names = [mname for mname, _ in experiments[0]["test_results"][cutoff].items()]
        num_metrics = len(metrics_names)
        cols = 3  # Number of columns for subplots
        rows = (num_metrics + cols - 1) // cols  # Calculate required number of rows

        data = []
        explorable_parameters = ["nbits", "neighbors", "ntables"]
        fixed_parameters = []
        variable_parameter = None
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        model_dict = config["experiment"]["models"][model_name]
        similarity = model_dict["similarity"]
        neighbors = model_dict.get("neighbors", None)
        model_name_full = model_name + "::" + similarity
        if similarity in ["rp_faiss", "rp_custom", "rp_hashtables", "rp_custommp", "rp_faisslike"]:
            for exp_param in explorable_parameters:
                value = model_dict.get(exp_param, None)
                if isinstance(value, list):
                    variable_parameter = exp_param
                else:
                    fixed_parameters.append(exp_param)

        param_values = []
        metrics_dict = {metric: [] for metric in metrics_names}

        # Gather data from experiments
        for experiment in experiments:
            param_values.append(experiment["params"][variable_parameter])
            experiment_data = {
                'Model': model_name_full,
                fixed_parameters[0]: experiment['params'][fixed_parameters[0]],
                fixed_parameters[1]: experiment['params'][fixed_parameters[1]],
                variable_parameter: experiment['params'][variable_parameter]
            }
            for metric in list(experiment["test_results"].values())[0].items():
                metrics_dict[metric[0]].append(metric[1])
                experiment_data[metric[0]] = metric[1]
            data.append(experiment_data)

        # Sort values based on the parameter values
        param_values_indices = np.argsort(param_values)
        ordered_param_values = np.asarray(param_values)[param_values_indices]

        # Generate directory name for current model with date
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_results_dir = os.path.join(final_path, f"{model_name_full}_{today}")
        os.makedirs(model_results_dir, exist_ok=True)

        # Plot each metric in a different subplot and save the final plot
        for idx, metric in enumerate(metrics_names):
            ax = axes[idx]
            ordered_values = np.asarray(metrics_dict[metric])[param_values_indices]
            ax.plot(ordered_param_values, ordered_values, marker="o",
                    label=f"{model_name_full} ({fixed_parameters[0]}&{fixed_parameters[1]}={experiment['params'][fixed_parameters[0]]}&{experiment['params'][fixed_parameters[1]]})")

            # Annotate percentage changes
            for i in range(1, len(ordered_param_values)):
                percent_change = ((ordered_values[i] - ordered_values[i - 1]) / ordered_values[i - 1]) * 100
                ax.annotate(f'{percent_change:.1f}%', (ordered_param_values[i], ordered_values[i]),
                            textcoords="offset points", xytext=(0, 10), ha='center')

            ax.set_xlabel(variable_parameter)
            ax.set_ylabel(metric)
            ax.legend()

        # Adjust layout and show plot
        plot_path = os.path.join(model_results_dir, f"{model_name_full}_{variable_parameter}_{neighbors}_{today}.png")
        plt.savefig(plot_path)
        plt.tight_layout()
        plt.show()

        # Convert list to DataFrame
        results_df = pd.DataFrame(data)
        # Sort DataFrame by 'Model' and 'Parameter Value'
        results_df.sort_values(by=['Model', variable_parameter], inplace=True)

        # Calculate percentage changes for each metric within each model
        for metric in metrics_names:
            results_df[metric + '_pct_change'] = results_df.groupby('Model')[metric].pct_change() * 100

        # Save DataFrame to CSV
        if print_csv:
            excel_path = os.path.join(model_results_dir,
                                      f"{model_name_full}_{variable_parameter}_{neighbors}_{today}_results.xlsx")
            csv_path = os.path.join(model_results_dir,
                                    f"{model_name_full}_{variable_parameter}_{neighbors}_{today}_results.csv")
            results_df.to_csv(csv_path, index=False)
            results_df.to_excel(excel_path, index=False)


def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points.
    For a point to be Pareto efficient, there should be no other point that is strictly better in both objectives
    (i.e., lower NDCG Loss Percent and higher Similarity Time Decrease Percent).
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    costs = np.asarray(list((map(lambda x: [x[0], -x[1]], costs))))
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Check if any other point is better in all dimensions
            is_efficient[i] = not np.any(np.all(costs < c, axis=1))
    return is_efficient


def create_color_map(unique_combinations):
    num_colors = len(unique_combinations)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # Use 'tab20' which has 20 colors
    if num_colors > 20:
        # Extend the palette if more than 20 colors are needed
        colors = np.vstack((colors, plt.cm.tab20b(np.linspace(0, 1, num_colors - 20))))
    color_map = {idx: colors[i % len(colors)] for i, idx in enumerate(unique_combinations.index)}
    return color_map


def plot_pareto_frontier(experiments_data, pareto_efficient_points, unique_combinations, color_map, plot_dir):
    plt.figure(figsize=(20, 10))  # Further increased width to 20
    for i, row in experiments_data.iterrows():
        color = color_map[unique_combinations[
            (unique_combinations['nbits'] == row['nbits']) & (unique_combinations['ntables'] == row['ntables'])].index[
            0]]
        legend_texts = [text.get_text() for text in
                        plt.gca().get_legend().get_texts()] if plt.gca().get_legend() else []
        plt.scatter(row['similarity_time_decrease_percent'], row['NDCG_loss_percent'], color=color,
                    label=f'nbits={row["nbits"]}, ntables={row["ntables"]}' if f'nbits={row["nbits"]}, ntables={row["ntables"]}' not in legend_texts else "")

    # Highlight Pareto efficient points with a border
    for _, row in pareto_efficient_points.iterrows():
        color = color_map[unique_combinations[
            (unique_combinations['nbits'] == row['nbits']) & (unique_combinations['ntables'] == row['ntables'])].index[
            0]]
        legend_texts = [text.get_text() for text in
                        plt.gca().get_legend().get_texts()] if plt.gca().get_legend() else []
        plt.scatter(row['similarity_time_decrease_percent'], row['NDCG_loss_percent'], edgecolor='black',
                    facecolor=color, s=100, linewidth=1.5,
                    label=f'Pareto nbits={row["nbits"]}, ntables={row["ntables"]}' if f'Pareto nbits={row["nbits"]}, ntables={row["ntables"]}' not in legend_texts else "")

    plt.axhline(0, color='gray', lw=0.5)  # NDCG loss 0 line
    plt.axvline(0, color='green', linestyle='--', label='No Similarity Time Change')
    plt.xlabel('Similarity Time Decrease (%)')
    plt.ylabel('NDCG Loss (%)')
    plt.title('Experiments vs. Baseline')

    if len(unique_combinations) <= 12:
        plt.legend(title="Configurations", bbox_to_anchor=(1.15, 1), loc='upper left')
    else:
        plt.legend(title="Configurations", loc='center left', bbox_to_anchor=(1.15, 0.5), ncol=2)

    plot_name = "scatter"

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1] if len(unique_combinations) > 12 else [0, 0, 1, 1])
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.show()


def plot_pareto_only(pareto_efficient_points, unique_combinations, color_map, plot_dir):
    plt.figure(figsize=(20, 10))  # Further increased width to 20
    for _, row in pareto_efficient_points.iterrows():
        color = color_map[unique_combinations[
            (unique_combinations['nbits'] == row['nbits']) & (unique_combinations['ntables'] == row['ntables'])].index[
            0]]
        plt.scatter(row['similarity_time_decrease_percent'], row['NDCG_loss_percent'], edgecolor='black',
                    facecolor=color, s=100, linewidth=1.5,
                    label=f'Pareto nbits={row["nbits"]}, ntables={row["ntables"]}')

    plt.axhline(0, color='gray', lw=0.5)  # NDCG loss 0 line
    plt.axvline(0, color='green', linestyle='--', label='No Similarity Time Change')
    plt.xlabel('Similarity Time Decrease (%)')
    plt.ylabel('NDCG Loss (%)')
    plt.title('Pareto Efficient Points')

    plot_name = "pareto_scatter"

    plt.legend(title="Configurations", loc='center left', bbox_to_anchor=(1.15, 0.5), ncol=2)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.show()


def plot_pareto_bars(experiments_data, color_map, unique_combinations, pareto_efficient_points, value_column, ylabel,
                     title, plot_dir):
    pareto_data = experiments_data.loc[pareto_efficient_points.index]
    sorted_data = pareto_data.sort_values(by=value_column, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(24, 12))

    bars = ax.bar(sorted_data.index, sorted_data[value_column],
                  color=[color_map[unique_combinations[
                      (unique_combinations['nbits'] == row['nbits']) & (
                              unique_combinations['ntables'] == row['ntables'])].index[0]]
                         for _, row in sorted_data.iterrows()])

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Pareto Efficient Experiments')

    # Annotate bars
    for bar, (_, row) in zip(bars, sorted_data.iterrows()):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{round(yval, 2)}%", ha='center', va='bottom')

    # Add title
    ax.set_title(title)

    # Create custom legend elements for Pareto points
    legend_elements = []
    for idx, row in pareto_efficient_points.iterrows():
        legend_elements.append(plt.Line2D([0], [0], color=color_map[unique_combinations[
            (unique_combinations['nbits'] == row['nbits']) & (
                    unique_combinations['ntables'] == row['ntables'])].index[0]], lw=4,
                                          label=f'Pareto nbits={row["nbits"]}, ntables={row["ntables"]}'))

    # Adding legend to the figure
    plot_name = "pareto_bars"
    if "similarity" in title.lower().strip(""):
        plot_name += "similarity"
    else:
        plot_name += "ndcg_loss"
    ax.legend(handles=legend_elements, title="Pareto Configurations", loc='center left', bbox_to_anchor=(1.02, 0.5),
              ncol=2)
    plt.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.show()


def plot_similarity_time_bar(experiments_data, color_map, unique_combinations, plot_dir):
    sorted_data = experiments_data.sort_values(by='similarity_time_decrease_percent', ascending=False).reset_index(
        drop=True)
    num_experiments = len(sorted_data)
    half_point = num_experiments // 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharey=True)

    # Plot first half of the data
    bars1 = ax1.bar(sorted_data.index[:half_point], sorted_data['similarity_time_decrease_percent'][:half_point],
                    color=[color_map[unique_combinations[
                        (unique_combinations['nbits'] == row['nbits']) & (
                                unique_combinations['ntables'] == row['ntables'])].index[0]]
                           for _, row in sorted_data.iloc[:half_point].iterrows()])

    # Plot second half of the data
    bars2 = ax2.bar(sorted_data.index[half_point:], sorted_data['similarity_time_decrease_percent'][half_point:],
                    color=[color_map[unique_combinations[
                        (unique_combinations['nbits'] == row['nbits']) & (
                                unique_combinations['ntables'] == row['ntables'])].index[0]]
                           for _, row in sorted_data.iloc[half_point:].iterrows()])

    ax1.set_ylabel('Similarity Time Decrease (%)')
    ax2.set_ylabel('Similarity Time Decrease (%)')
    ax2.set_xlabel('Experiments')

    # Annotate bars
    for bars, ax in zip([bars1, bars2], [ax1, ax2]):
        for bar, (_, row) in zip(bars, sorted_data.iterrows()):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{round(yval, 2)}%", ha='center', va='bottom')

    # Add titles
    ax1.set_title('Similarity Time Decrease Comparison - First Half')
    ax2.set_title('Similarity Time Decrease Comparison - Second Half')

    # Create custom legend elements
    legend_elements = []
    for idx, row in unique_combinations.iterrows():
        legend_elements.append(plt.Line2D([0], [0], color=color_map[idx], lw=4,
                                          label=f'nbits={row["nbits"]}, ntables={row["ntables"]}'))

    # Adding legend to the figure
    plot_name = "similarity_bar"
    ax1.legend(handles=legend_elements, title="Configurations", loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=2)
    plt.subplots_adjust(right=0.75, hspace=0.5)
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.show()


def plot_ndcg_loss_bar(experiments_data, color_map, unique_combinations, plot_dir):
    sorted_data = experiments_data.sort_values(by='NDCG_loss_percent', ascending=False).reset_index(drop=True)
    num_experiments = len(sorted_data)
    half_point = num_experiments // 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharey=True)

    # Plot first half of the data
    bars1 = ax1.bar(sorted_data.index[:half_point], sorted_data['NDCG_loss_percent'][:half_point],
                    color=[color_map[unique_combinations[
                        (unique_combinations['nbits'] == row['nbits']) & (
                                unique_combinations['ntables'] == row['ntables'])].index[0]]
                           for _, row in sorted_data.iloc[:half_point].iterrows()])

    # Plot second half of the data
    bars2 = ax2.bar(sorted_data.index[half_point:], sorted_data['NDCG_loss_percent'][half_point:],
                    color=[color_map[unique_combinations[
                        (unique_combinations['nbits'] == row['nbits']) & (
                                unique_combinations['ntables'] == row['ntables'])].index[0]]
                           for _, row in sorted_data.iloc[half_point:].iterrows()])

    ax1.set_ylabel('NDCG Loss Percent')
    ax2.set_ylabel('NDCG Loss Percent')
    ax2.set_xlabel('Experiments')

    # Annotate bars
    for bars, ax in zip([bars1, bars2], [ax1, ax2]):
        for bar, (_, row) in zip(bars, sorted_data.iterrows()):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{round(yval, 2)}%", ha='center', va='bottom')

    # Add titles
    ax1.set_title('NDCG Loss Comparison - First Half')
    ax2.set_title('NDCG Loss Comparison - Second Half')

    # Create custom legend elements
    legend_elements = []
    for idx, row in unique_combinations.iterrows():
        legend_elements.append(plt.Line2D([0], [0], color=color_map[idx], lw=4,
                                          label=f'nbits={row["nbits"]}, ntables={row["ntables"]}'))

    # Adding legend to the figure
    plot_name = "ndgc_loss"
    ax1.legend(handles=legend_elements, title="Configurations", loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=2)
    plt.subplots_adjust(right=0.75, hspace=0.5)
    plt.savefig(os.path.join(plot_dir, plot_name))
    plt.show()


def print_pareto_details(pareto_efficient_points, baseline_data, plot_dir):
    print("\nPareto Efficient Experiments:")
    for idx, row in pareto_efficient_points.iterrows():
        print(f"Model: {row['model']}")
        print(f"  nDCG Rendle 2020: {row['nDCGRendle2020']}")
        print(f"  Similarity Time: {row['similarity_time']} seconds")
        print(f"  NDCG Loss: {row['NDCG_loss']}")
        print(f"  NDCG Loss Percent: {row['NDCG_loss_percent']}%")
        print(f"  Similarity Time Decrease Percent: {row['similarity_time_decrease_percent']}%")
        print(f"  nbits: {row['nbits']}")
        print(f"  ntables: {row['ntables']}")
        print("-" * 40)
    file_name = "pareto_points.csv"
    print("\nBaseline Details:")
    print(f"Model: {baseline_data['model'].iloc[0]}")
    print(f"  nDCG Rendle 2020: {baseline_data['nDCGRendle2020'].iloc[0]}")
    print(f"  Similarity Time: {baseline_data['similarity_time'].iloc[0]} seconds")
    pareto_efficient_points.to_csv(os.path.join(plot_dir, file_name))


def find_best_experiment_static(baseline_path, experiments_path):
    # Load the data
    baseline_data = pd.read_csv(baseline_path, sep='\t')
    experiments_data = pd.read_csv(experiments_path, sep='\t')

    base_path = "/".join(baseline_path.split("/")[:-1], )

    dir_name = experiments_path.split("/")[-1].split(".")[0] + "_comparison"

    # Directory where all the plots will be saved
    plot_dir = os.path.join(base_path, dir_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Extract baseline metrics
    baseline_ndcg = baseline_data['nDCGRendle2020'].iloc[0]
    baseline_similarity_time = baseline_data['similarity_time'].iloc[0]

    # Calculate differences and adjust labels for percentage changes
    experiments_data['NDCG_loss'] = baseline_ndcg - experiments_data['nDCGRendle2020']
    experiments_data['similarity_time_difference'] = experiments_data['similarity_time'] - baseline_similarity_time
    experiments_data['NDCG_loss_percent'] = (experiments_data['NDCG_loss'] / baseline_ndcg) * 100
    experiments_data['similarity_time_decrease_percent'] = (-experiments_data[
        'similarity_time_difference'] / baseline_similarity_time) * 100

    index = 5
    # Parse nbits and ntables from model description
    if "ItemKNN" in experiments_data["model"][0]:
        index += 1
    experiments_data['nbits'] = [int(el.split("_")[index].split("=")[1]) for el in experiments_data["model"]]
    experiments_data['ntables'] = [int(el.split("_")[index + 1].split("=")[1]) for el in experiments_data["model"]]

    # Filter experiments to only include those with decreased similarity time
    valid_experiments = experiments_data[experiments_data['similarity_time'] < baseline_similarity_time]

    # Calculate Pareto efficient points
    costs = valid_experiments[['NDCG_loss_percent', 'similarity_time_decrease_percent']].values
    pareto_efficient_mask = is_pareto_efficient(costs)
    pareto_efficient_points = valid_experiments[pareto_efficient_mask]

    # Create a color map for each unique nbits and ntables combination
    unique_combinations = experiments_data[['nbits', 'ntables']].drop_duplicates().reset_index()
    color_map = create_color_map(unique_combinations)

    # Plotting results with matplotlib
    plot_pareto_frontier(experiments_data, pareto_efficient_points, unique_combinations, color_map, plot_dir)
    if len(unique_combinations) > 12:
        plot_pareto_only(pareto_efficient_points, unique_combinations, color_map, plot_dir)
    plot_similarity_time_bar(experiments_data, color_map, unique_combinations, plot_dir)
    plot_ndcg_loss_bar(experiments_data, color_map, unique_combinations, plot_dir)
    plot_pareto_bars(experiments_data, color_map, unique_combinations, pareto_efficient_points,
                     'similarity_time_decrease_percent', 'Similarity Time Decrease (%)',
                     'Pareto Efficient Similarity Time Decrease', plot_dir)
    plot_pareto_bars(experiments_data, color_map, unique_combinations, pareto_efficient_points,
                     'NDCG_loss_percent', 'NDCG Loss Percent', 'Pareto Efficient NDCG Loss', plot_dir)

    # Print comprehensive comparison and best experiment details
    print_pareto_details(pareto_efficient_points, baseline_data, plot_dir)

    return pareto_efficient_points


def compare_experiments(first_experiments_path, second_experiments_path, variable_parameter="nbits"):
    # Load the data
    first_experiments_data = pd.read_csv(first_experiments_path, sep='\t')
    second_experiments_data = pd.read_csv(second_experiments_path, sep='\t')

    first_experiment_model_name=first_experiments_data["model"][0].split("=")[2]
    second_experiment_model_name=second_experiments_data["model"][0].split("=")[2]

    metrics = first_experiments_data.columns[1:].tolist()
    if len(first_experiments_data) != len(second_experiments_data):
        raise Exception("The two dataframe should have the same shape")

    base_path = os.path.dirname(first_experiments_path)
    dir_name = os.path.join(base_path, first_experiments_path.split("/")[-1].split(".")[0] + "_" + \
                            second_experiments_path.split("/")[-1].split("_")[-1].split(".")[0] + "_comparison")

    # Create the directory if it does not exist
    os.makedirs(dir_name, exist_ok=True)

    # Expand and sort dataframes
    first_experiments_data = expand_results_dataframe(first_experiments_data).sort_values(by=variable_parameter)
    second_experiments_data = expand_results_dataframe(second_experiments_data).sort_values(by=variable_parameter)

    num_metrics = len(metrics)
    cols = 3  # Number of columns for subplots
    rows = (num_metrics + cols - 1) // cols  # Calculate required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    variable_values = first_experiments_data[variable_parameter].values

    # Plot each metric in a different subplot
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        first_values = first_experiments_data[metric].values
        second_values = second_experiments_data[metric].values

        ax.plot(variable_values, first_values, marker="o", label=f"{first_experiment_model_name}", color="blue")
        ax.plot(variable_values, second_values, marker="o", label=f"{second_experiment_model_name}t", color="orange")

        # Annotate percentage changes for first experiment
        for i in range(1, len(variable_values)):
            percent_change = ((first_values[i] - first_values[i - 1]) / first_values[i - 1]) * 100
            ax.annotate(f'{percent_change:.1f}%', (variable_values[i], first_values[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center', color='blue')

        # Annotate percentage changes for second experiment
        for i in range(1, len(variable_values)):
            percent_change = ((second_values[i] - second_values[i - 1]) / second_values[i - 1]) * 100
            ax.annotate(f'{percent_change:.1f}%', (variable_values[i], second_values[i]),
                        textcoords="offset points", xytext=(0, -15), ha='center', color='orange')

        ax.set_xlabel(variable_parameter)
        ax.set_ylabel(metric)
        ax.legend()

    # Hide any unused subplots
    for ax in axes[num_metrics:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, 'comparison_plot.png'))
    plt.show()


def average_column_from_subdirectories(main_directory, model="UserKNN", similarity="rp_faiss"):
    # Model forse non ci serve
    data_frames = []
    main_path = Path(main_directory)
    group_cols = ['Model', 'neighbors', 'ntables', 'nbits']
    columns_to_average = ['nDCGRendle2020', 'Recall', 'HR', 'Precision', 'MAP', 'MRR', 'similarity_time',
                          'indexing_time', 'candidates_retrieval_time', 'similarity_matrix_time']

    metrics_names = ['nDCGRendle2020', 'Recall', 'HR', 'Precision', 'MAP', 'MRR', 'similarity_time',
                     'indexing_time', 'candidates_retrieval_time', 'similarity_matrix_time']

    filtered_subdirectories = [subdirectory for subdirectory in main_path.iterdir() if
                               subdirectory.is_dir() and model in str(subdirectory) and similarity in str(
                                   subdirectory.name)]
    print(f"Model: {model}")
    print(f"Similarity: {similarity}")
    print(f"The average is computed over {len(filtered_subdirectories)} experiments")

    for subdirectory in filtered_subdirectories:
        # Find all CSV files in the subdirectory
        csv_files = list(subdirectory.glob('*.csv'))

        # Ensure there is exactly one CSV file in the subdirectory
        if len(csv_files) == 1:
            csv_file_path = csv_files[0]
            df = pd.read_csv(csv_file_path)[group_cols + metrics_names]
            data_frames.append(df)
        else:
            print(f"Warning: {subdirectory} contains {len(csv_files)} CSV files.")

    if not data_frames:
        raise ValueError("No valid CSV files found in the subdirectories.")

    # Concatenate all dataframes to ensure they have the same structure
    combined_df = pd.concat(data_frames)

    # Group by all columns except the specified one and compute the mean of the specified column
    averaged_df = combined_df.groupby(group_cols, as_index=False)[columns_to_average].mean()

    num_metrics = len(metrics_names)
    cols = 3  # Number of columns for subplots
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        ordered_values = averaged_df[metric].tolist()
        ordered_param_values = averaged_df["nbits"].tolist()  # For now it's hardcoded
        ax.plot(ordered_param_values, ordered_values, marker="o",
                label=model + similarity)
        # Annotate percentage changes
        for i in range(1, len(ordered_param_values)):
            percent_change = ((ordered_values[i] - ordered_values[i - 1]) / ordered_values[i - 1]) * 100
            ax.annotate(f'{percent_change:.1f}%', (ordered_param_values[i], ordered_values[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center')

        ax.set_xlabel("nbits")
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()
    plt.show()

    return averaged_df


def expand_results_dataframe(results_df: pd.DataFrame):
    index = 5
    # Parse nbits and ntables from model description
    if "ItemKNN" in results_df["model"][0]:
        index += 1
    results_df['nbits'] = [int(el.split("_")[index].split("=")[1]) for el in results_df["model"]]
    results_df['ntables'] = [int(el.split("_")[index + 1].split("=")[1]) for el in results_df["model"]]
    return results_df
