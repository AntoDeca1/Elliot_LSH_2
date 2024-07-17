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


def plot_pareto_frontier(experiments_data, pareto_efficient_points, unique_combinations, color_map, plot_dir,
                         best_point):
    plt.figure(figsize=(20, 10))  # Further increased width to 20

    pareto_models = set(pareto_efficient_points['model'])
    for i, row in experiments_data.iterrows():
        if row["model"] in pareto_models:
            continue
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
    if best_point is not None:
        # Highlight the best experiment with a cross
        plt.scatter(best_point['similarity_time_decrease_percent'], best_point['NDCG_loss_percent'], color='red',
                    marker='x', s=200, label=f'Best nbits={best_point["nbits"]}, ntables={best_point["ntables"]} ')

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


def plot_pareto_only(pareto_efficient_points, unique_combinations, color_map, plot_dir, best_point):
    plt.figure(figsize=(20, 10))  # Further increased width to 20
    for _, row in pareto_efficient_points.iterrows():
        color = color_map[unique_combinations[
            (unique_combinations['nbits'] == row['nbits']) & (unique_combinations['ntables'] == row['ntables'])].index[
            0]]
        plt.scatter(row['similarity_time_decrease_percent'], row['NDCG_loss_percent'], edgecolor='black',
                    facecolor=color, s=100, linewidth=1.5,
                    label=f'Pareto nbits={row["nbits"]}, ntables={row["ntables"]}')
    plt.scatter(best_point['similarity_time_decrease_percent'], best_point['NDCG_loss_percent'], color='red',
                marker='x', s=200, label=f'Best nbits={best_point["nbits"]}, ntables={best_point["ntables"]} ')

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
                     title, plot_dir, best_point):
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

    if best_point is not None:
        # Highlight the best experiment with a dotted border
        best_index = sorted_data.index[sorted_data['model'] == best_point['model']].tolist()[0]
        bars[best_index].set_edgecolor('red')
        bars[best_index].set_linewidth(2)
        bars[best_index].set_linestyle('dotted')

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


def print_pareto_details(pareto_efficient_points, baseline_data, plot_dir, best_point):
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

    if best_point is not None:
        print("\n Best Experiment")
        print(f"Model: {best_point['model']}")
        print(f"  nDCG Rendle 2020: {best_point['nDCGRendle2020']}")
        print(f"  Similarity Time: {best_point['similarity_time']} seconds")
        print(f"  NDCG Loss: {best_point['NDCG_loss']}")
        print(f"  NDCG Loss Percent: {best_point['NDCG_loss_percent']}%")
        print(f"  Similarity Time Decrease Percent: {best_point['similarity_time_decrease_percent']}%")
        print(f"  nbits: {best_point['nbits']}")
        print(f"  ntables: {best_point['ntables']}")
        print("-" * 40)

    file_name = "pareto_points.csv"
    print("\nBaseline Details:")
    print(f"Model: {baseline_data['model'].iloc[0]}")
    print(f"  nDCG Rendle 2020: {baseline_data['nDCGRendle2020'].iloc[0]}")
    print(f"  Similarity Time: {baseline_data['similarity_time'].iloc[0]} seconds")
    pareto_efficient_points.to_csv(os.path.join(plot_dir, file_name))


def plot_fairness(baseline_data, experiments_data, fairness_cols, color_map, plot_dir, unique_combinations, best_point):
    """
    Useful function for plotting comparisons between baseline and fairness metrics. It creates a plot for each user group and shows the bias compared to the baseline.
    In case the metric of interest is "BiasDisparityBR", if the "BiasDisparityBS" is in our results it's added to the comparison
    :param baseline_data:
    :param experiments_data:
    :param fairness_cols:
    :param color_map:
    :param plot_dir:
    :param unique_combinations:
    :return:
    """
    baseline_biases = baseline_data[["model"] + fairness_cols]
    experiments_biases = experiments_data[["model", "nbits", "ntables"] + fairness_cols]

    metric_name = fairness_cols[0].split("_")[0]
    fairness_dir = os.path.join(plot_dir, "fairness", metric_name)
    os.makedirs(fairness_dir, exist_ok=True)

    if metric_name in ["BiasDisparityBR", "BiasDisparityBD"]:
        plot_Bias_Disparity(baseline_data, baseline_biases, experiments_biases, fairness_cols, metric_name,
                            unique_combinations, color_map, fairness_dir, best_point)
    else:
        plot_RankingOppurtinity(baseline_biases, experiments_biases, fairness_cols, metric_name,
                                unique_combinations, color_map, fairness_dir, best_point)


def plot_Bias_Disparity(baseline_data, baseline_biases, experiments_biases, fairness_cols, metric_name,
                        unique_combinations, color_map, fairness_dir, best_point):
    user_groups = set()
    item_groups = set()

    # Rename the original column names in a more readable format
    user_groups, item_groups, baseline_biases, experiments_biases = rename_dataframe_fairness_cols(baseline_biases,
                                                                                                   experiments_biases,
                                                                                                   fairness_cols,
                                                                                                   item_groups,
                                                                                                   user_groups)

    bias_source_df = pd.DataFrame()
    if metric_name == "BiasDisparityBR":
        # Check for BiasDisparityBS columns
        bias_disparity_bs_cols = [col for col in baseline_data.columns if "BiasDisparityBS" in col]
        if len(bias_disparity_bs_cols) > 0:
            bias_source_model = {"model": "BiasSource"}
            for col in bias_disparity_bs_cols:
                user_group, item_group = col.split(":")[-2:]
                user_group_id = int(user_group.split("-")[1][0])
                item_group_id = int(item_group[-1])
                bias_source_model[f"{user_group_id}:{item_group_id}"] = baseline_data[col].iloc[0]
            bias_source_df = pd.DataFrame([bias_source_model])

    user_groups = sorted(user_groups)
    item_groups = sorted(item_groups)
    combined_data = pd.concat([bias_source_df, baseline_biases, experiments_biases])
    default_color = 'grey'

    # Create a plot for each user group
    for user_group in user_groups:
        plot_grouped_bars(combined_data, unique_combinations, color_map, fairness_dir, metric_name, best_point,
                          item_groups,
                          user_group)


def plot_grouped_bars(combined_data, unique_combinations, color_map, fairness_dir, metric_name, best_point,
                      item_groups=None,
                      user_group=None,
                      default_color="grey", ):
    """
    Function for plotting grouped bars plots
    Used for FairnessMetrics(BiasDisparityBR,BiasDisparityBD,REO,RSP)
    Used for other additional metrics we consider in find_experiment_static() (e.g Gini,ItemCoverage,ShannonEntropy....)
    :param combined_data: dataframe containing both baseline and experiments results concatenated
    :param unique_combinations: unique combinations of lsh parameters(nbits and ntables)
    :param color_map: lookup table containing the color for each combination
    :param fairness_dir: directory in witch we results will be saved
    :param metric_name:
    :param item_groups: set containg the item group ids
    :param user_group: set containing the user group ids
    :param default_color: color to assign to the baseline
    :return:
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    bar_width = 1.5 / len(combined_data)  # Make bars narrower to add space within groups
    group_spacing = 2.0  # Add more space between different item groups
    intra_group_spacing = 0  # Add space between bars within each group

    if item_groups is not None:
        x = np.arange(len(item_groups)) * (1 + group_spacing)  # Adjust x to include spacing between groups
    else:
        x = np.arange(1)
    num_bars = len(combined_data)

    if user_group is not None:
        user_group = f"{user_group}:"
    else:
        user_group = ""

    for i, (_, row) in enumerate(combined_data.iterrows()):
        if item_groups is not None:
            biases = [row[f"{user_group}{item_group}"] if f"{user_group}{item_group}" in row else 0 for item_group in
                      item_groups]
        else:
            biases = [row[metric_name]]

        if np.isnan(row["nbits"]) or np.isnan(row["ntables"]):
            if row["model"] == "BiasSource":
                color = "red"
                label = 'bias_source'
            else:
                color = default_color
                label = 'baseline'
        else:
            color_index = unique_combinations[(unique_combinations["nbits"] == row["nbits"]) & (
                    unique_combinations["ntables"] == row["ntables"])].index[0]
            color = color_map.get(color_index, default_color)
            label = f'nbits={row["nbits"]}, ntables={row["ntables"]}'

        bar_positions = x + (i - num_bars / 2) * (bar_width + intra_group_spacing)  # Add intra-group spacing
        bars = ax.bar(bar_positions, biases, bar_width, label=label, color=color)

        # for j, bias in enumerate(biases):
        #     ax.text(bar_positions[j], bias, f"{bias:.3f}", ha='center', va='bottom')  # Show 3 decimal places

        # Highlight the best experiment with a dotted border
        if best_point is not None:
            if row['model'] == best_point['model']:
                for bar in bars:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2)
                    bar.set_linestyle('dotted')

    ax.set_ylabel(f'{metric_name}')
    if user_group != "":
        ax.set_title(f'Metric: {metric_name} for User Group {user_group}')
    else:
        ax.set_title(f'Metric: {metric_name} ')
    ax.set_xticks(x)
    if item_groups is not None:
        ax.set_xlabel('Item Groups')
        ax.set_xticklabels([f"Item {item_group}" for item_group in item_groups])
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Configurations", loc='upper left',
              bbox_to_anchor=(1, 1))

    plot_name = f"{metric_name}_group_{user_group}.png"
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fairness_dir, plot_name))
    plt.show()


def rename_dataframe_fairness_cols(baseline_biases, experiments_biases, fairness_cols, item_groups=None,
                                   user_groups=None):
    """
    Utility function for renaming the original dataframe columns(taken from Elliot tsv results) in a more readable format
    We also keep track of:
    - Number of users groups
    - Number of items groups

    For metrics like BiasDisparity we have both user and item groups
    For metrics like REO,RSP we have just items_groups
    :param baseline_biases: dataframe composed only by model_name + fairness_columns
    :param experiments_biases: dataframe composed only by model_name + fairness_columns
    :param fairness_cols: cols containing the value of the metrics we are considering
    :param item_groups: set containing the ids of the items groups
    :param user_groups: set containing the ids of the users groups
    :return:
    """
    if (item_groups is not None) and (user_groups is not None):
        for col in fairness_cols:
            user_group, item_group = col.split(":")[-2:]
            user_group_id = int(user_group.split("-")[1][0])
            item_group_id = int(item_group[-1])
            if col in baseline_biases.columns:
                baseline_biases = baseline_biases.rename(columns={col: f"{user_group_id}:{item_group_id}"})
            if col in experiments_biases.columns:
                experiments_biases = experiments_biases.rename(columns={col: f"{user_group_id}:{item_group_id}"})
            user_groups.add(user_group_id)
            item_groups.add(item_group_id)
        return user_groups, item_groups, baseline_biases, experiments_biases
    elif item_groups is not None:
        for col in fairness_cols:
            if "ProbToBeRanked" in col:
                _, item_group = col.split(":")[-2:]
                item_group_id = int(item_group[-1])
                if col in baseline_biases.columns:
                    baseline_biases = baseline_biases.rename(columns={col: f"{item_group_id}"})
                if col in experiments_biases.columns:
                    experiments_biases = experiments_biases.rename(columns={col: f"{item_group_id}"})
                item_groups.add(item_group_id)
            else:
                new_name = col.split("_")[0]
                if col in baseline_biases.columns:
                    baseline_biases = baseline_biases.rename(columns={col: f"{new_name}"})
                if col in experiments_biases.columns:
                    experiments_biases = experiments_biases.rename(columns={col: f"{new_name}"})
        return item_groups, baseline_biases, experiments_biases
    else:
        raise Exception("Missing user_groups or item groups set")


def plot_RankingOppurtinity(baseline_biases, experiments_biases, fairness_cols, metric_name,
                            unique_combinations, color_map, fairness_dir, best_point):
    item_groups = set()

    # Rename the original column names in a more readable format
    item_groups, baseline_biases, experiments_biases = rename_dataframe_fairness_cols(baseline_biases,
                                                                                      experiments_biases, fairness_cols,
                                                                                      item_groups)
    item_groups = sorted(item_groups)
    combined_data = pd.concat([baseline_biases, experiments_biases])
    default_color = 'grey'

    # Plot showing the Ranking Bias for each item category
    plot_grouped_bars(combined_data, unique_combinations, color_map, fairness_dir, metric_name, best_point, item_groups)

    # Plot showing the final Ranking Bias
    single_metric_name = metric_name.split("-")[0]
    plot_grouped_bars(combined_data, unique_combinations, color_map, fairness_dir, metric_name=single_metric_name,
                      best_point=best_point,
                      item_groups=None)


def find_best_experiment_static(baseline_path, experiments_path):
    """
    Function that given the path where the baseline results are saved and the path where the lsh experiments are saved
    1) Compare each single experiment with the baseline
    2) Create a scatter plot showing the similarity decrease and ndcg loss with respect to the baseline
    3) Filter the previour scatter plot showing only the pareto efficient points(for better visualization)
    !----------------NEW----------------------------
    4) Highlight the best experiments with a red dotted border (the one that has the greatest gap between Similarity decrease and NDCG Loss)
    5) Integrate the possibility of comparing Fairness matrics(BiasDisparityBR,BiasDisparityBS,BiasDisparityBD,REO,RSP)
    6) Integrate the possibility of comparing novelty and diversity metrics
    :param baseline_path:
    :param experiments_path:
    :return:
    """
    # Load the data
    baseline_data = pd.read_csv(baseline_path, sep='\t')
    experiments_data = pd.read_csv(experiments_path, sep='\t')

    # base path where the results will be saved
    base_path = "/".join(experiments_path.split("/")[:-1], )
    # name of the directory where the results will be saved in
    dir_name = experiments_path.split("/")[-1].split(".")[0] + "_comparison"

    # Full path for the results
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

    # Idea: Select the best point the one that has the greatest gap between similarity decrease and ndcg_loss_percent
    if len(costs) != 0:
        best_point_index = np.argmax([similarity_decrease - ndcg_loss for ndcg_loss, similarity_decrease in costs])
        best_point = valid_experiments.iloc[best_point_index]
    else:
        best_point = None

    # Create a color map for each unique nbits and ntables combination
    unique_combinations = experiments_data[['nbits', 'ntables']].drop_duplicates().reset_index()
    color_map = create_color_map(unique_combinations)

    # Plotting results
    plot_pareto_frontier(experiments_data, pareto_efficient_points, unique_combinations, color_map, plot_dir,
                         best_point)
    if len(unique_combinations) > 12:
        plot_pareto_only(pareto_efficient_points, unique_combinations, color_map, plot_dir, best_point)
    plot_similarity_time_bar(experiments_data, color_map, unique_combinations, plot_dir)
    plot_ndcg_loss_bar(experiments_data, color_map, unique_combinations, plot_dir)
    plot_pareto_bars(experiments_data, color_map, unique_combinations, pareto_efficient_points,
                     'similarity_time_decrease_percent', 'Similarity Time Decrease (%)',
                     'Pareto Efficient Similarity Time Decrease', plot_dir, best_point)
    plot_pareto_bars(experiments_data, color_map, unique_combinations, pareto_efficient_points,
                     'NDCG_loss_percent', 'NDCG Loss Percent', 'Pareto Efficient NDCG Loss', plot_dir, best_point)

    # Print comprehensive comparison and best experiment details
    print_pareto_details(pareto_efficient_points, baseline_data, plot_dir, best_point)

    # Fairness metrics analysis
    # BiasDisparityBS is independent from the model and so is added as extra BAR in the plot if present
    # "BiasDisparityBD"
    fairness_metrics = ["BiasDisparityBR", "BiasDisparityBD", "REO", "RSP"]
    for f_metric in fairness_metrics:
        baseline_fmetric_cols = [col for col in baseline_data.columns if f_metric in col]
        experiment_fmetric_cols = [col for col in experiments_data.columns if f_metric in col]
        n_baseline_fmetric_cols = len(baseline_fmetric_cols)
        n_experiment_fmetric_cols = len(experiment_fmetric_cols)
        if n_baseline_fmetric_cols != 0 and n_experiment_fmetric_cols != 0 and (
                n_baseline_fmetric_cols == n_experiment_fmetric_cols):
            plot_fairness(baseline_data, pareto_efficient_points, baseline_fmetric_cols, color_map, plot_dir,
                          unique_combinations, best_point)
    single_metrics = ["Gini", "SEntropy", "ItemCoverage", "EFD", "EPC"]
    for f_metric in single_metrics:
        if (f_metric in baseline_data.columns) and (f_metric in experiments_data.columns):
            plot_single_metric(baseline_data, pareto_efficient_points, [f_metric], color_map, plot_dir,
                               unique_combinations, best_point)

    return pareto_efficient_points


def plot_single_metric(baseline_data, experiments_data, metric_col, color_map, plot_dir, unique_combinations,
                       best_point):
    baseline_biases = baseline_data[["model"] + metric_col]
    experiments_biases = experiments_data[["model", "nbits", "ntables"] + metric_col]
    metric_name = metric_col[0]
    output_dir = os.path.join(plot_dir, metric_name)
    os.makedirs(output_dir, exist_ok=True)
    combined_data = pd.concat([baseline_biases, experiments_biases])
    plot_grouped_bars(combined_data, unique_combinations, color_map, output_dir, metric_name, best_point)


def compare_experiments(first_experiments_path, second_experiments_path, variable_parameter="nbits"):
    # Load the data
    first_experiments_data = pd.read_csv(first_experiments_path, sep='\t')
    second_experiments_data = pd.read_csv(second_experiments_path, sep='\t')

    first_experiment_model_name = first_experiments_data["model"][0].split("=")[2]
    second_experiment_model_name = second_experiments_data["model"][0].split("=")[2]

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
