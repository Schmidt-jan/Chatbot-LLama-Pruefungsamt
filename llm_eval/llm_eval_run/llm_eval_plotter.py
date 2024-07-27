from calendar import c
import os
import json
import sys
from turtle import color
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patheffects import withStroke
#from scipy.stats import linregress

HTWG_COLOR_1 = "#009b91"

# Global variables for color interpolation
PLOT_COLOR_START = "#009b91"#009B91
PLOT_COLOR_END = "#009b91" #275861

# Maximum number of test runs to display in timeline plots
TIMELINE_MAX_LENGTH = 5

promptfoo_runs = {
    #"20240628_101349": "TestRun 2",
    "20240627_205930": "CompleteRun",
    #"20240628_164154": "TestRun 3",
    #"20240629_224248": "BenniRun"
    "20240630_113545": "BenniRunFinetune1",
    "20240630_220713": "BenniRunFinetune2",
    "20240701_151846": "BenniRunFinetune3"
}

PLOT_TYPE = 'boxplot'    # 'boxplot' or 'violinplot'
FILE_TYPE = 'png'             # 'png' or 'svg'

def create_interpolated_palette(start_color, end_color, n_colors):
    cmap = LinearSegmentedColormap.from_list("custom_palette", [start_color, end_color], N=n_colors)
    palette = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return palette

def main(timestamp):
    base_output_path = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output'

    # Define output directories
    raw_data_path = create_directory(base_output_path + '/promptfoo_data/raw')
    processed_data_path = create_directory(base_output_path + '/promptfoo_data/processed')
    component_results_path = create_directory(processed_data_path + '/componentResults')
    generation_results_path = create_directory(processed_data_path + '/generationResults')
    
    plots_path = create_directory(base_output_path + '/plots')
    current_plots_path = create_directory(plots_path + f'/{timestamp}')
    conclusions_metrics_path = create_directory(current_plots_path + '/conclusions/metrics')
    conclusions_models_path = create_directory(current_plots_path + '/conclusions/models')
    singles_metrics_path = create_directory(current_plots_path + '/singles/metrics')
    singles_models_path = create_directory(current_plots_path + '/singles/models')

    conclusions_metrics_singles_path = create_directory(conclusions_metrics_path + '/singles')
    conclusions_models_singles_path = create_directory(conclusions_models_path + '/singles')

    conclusions_metrics_combined_path = create_directory(conclusions_metrics_path + '/combined')
    conclusions_models_combined_path = create_directory(conclusions_models_path + '/combined')

    input_file_cR = f'{component_results_path}/output_cR_{timestamp}.json'


    generate_bar_plots_perMetric(singles_metrics_path, input_file_cR)
    generate_bar_plots_perModel(singles_models_path, input_file_cR)

    create_plots_conclusions_perMetric(conclusions_metrics_combined_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    create_plots_conclusions_perModel(conclusions_models_combined_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)

    create_plots_conclusions_perMetric_singles(conclusions_metrics_singles_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    create_plots_conclusions_perModel_singles(conclusions_models_singles_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")
    return path

def save_plot(data, x, y, title, xlabel, save_path, palette, figsize=(6, 10)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x, y=y, palette=palette, width=0.5, fliersize=0, hue=y, dodge=False)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.tight_layout()

    # Save plot
    print(f"Saving plot to '{os.path.basename(save_path)}'")
    plt.savefig(save_path)
    plt.close()


def generate_bar_plots_perMetric(output_directory, input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame([{
        'Model': item['metric']['model'],
        'Metric': item['metric']['name'],
        'Similarity Score': item['score'],
        'Specificity': item['metric']['testContext']['vars']['quality']['specificity'],
        'Relevance': item['metric']['testContext']['vars']['quality']['relevance']
    } for item in data])

    df = df.sort_values(by='Model')
    unique_metrics = sorted(df['Metric'].unique())
    unique_models = sorted(df['Model'].unique())

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_models))

    for metric in unique_metrics:
        metric_data = df[df['Metric'] == metric]

        # Plot for Similarity Score
        save_plot(metric_data, 'Similarity Score', 'Model',
                  f'Similarity Scores Models - {metric}',
                  'Metric Score',
                  os.path.join(output_directory, f'singles_metrics_{metric}_similarity.png'),
                  figsize=(10, len(unique_models) * 0.5), palette=palette)

        # Weighted scores
        weighted_scores = {
            'Specificity_Relevance': metric_data['Similarity Score'] * (1 - metric_data['Specificity']) * metric_data[
                'Relevance'],
            'Specificity': metric_data['Similarity Score'] * (1 - metric_data['Specificity']),
            'Relevance': metric_data['Similarity Score'] * metric_data['Relevance']
        }

        for score_name, score_values in weighted_scores.items():
            save_plot(metric_data.assign(WeightedScore=score_values), 'WeightedScore', 'Model',
                      f'Weighted [{score_name.capitalize()}] - {metric}',
                      f'Weighted [{score_name.capitalize()}] Score',
                      os.path.join(output_directory, f'singles_metrics_{metric}_{score_name}.png'),
                      figsize=(10, len(unique_models) * 0.5), palette=palette)

def generate_bar_plots_perModel(output_directory, input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame([{
        'Model': item['metric']['model'],
        'Metric': item['metric']['name'],
        'Similarity Score': item['score'],
        'Specificity': item['metric']['testContext']['vars']['quality']['specificity'],
        'Relevance': item['metric']['testContext']['vars']['quality']['relevance']
    } for item in data])

    df = df.sort_values(by='Model')
    unique_models = sorted(df['Model'].unique())
    unique_metrics = sorted(df['Metric'].unique())

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_metrics))

    for model in unique_models:
        model_data = df[df['Model'] == model]

        # Plot for Similarity Score
        save_plot(model_data, 'Similarity Score', 'Metric',
                  f'Similarity Scores Metrics - {model}',
                  'Metric Score',
                  os.path.join(output_directory, f'singles_models_{model}_similarity.png'),
                  figsize=(10, len(unique_metrics) * 0.5), palette=palette)

        # Weighted scores
        weighted_scores = {
            'Specificity_Relevance': model_data['Similarity Score'] * (1 - model_data['Specificity']) * model_data['Relevance'],
            'Specificity': model_data['Similarity Score'] * (1 - model_data['Specificity']),
            'Relevance': model_data['Similarity Score'] * model_data['Relevance']
        }

        for score_name, score_values in weighted_scores.items():
            save_plot(model_data.assign(WeightedScore=score_values), 'WeightedScore', 'Metric',
                      f'Weighted [{score_name.capitalize()}] - {model}',
                      f'Weighted [{score_name.capitalize()}] Score',
                      os.path.join(output_directory, f'singles_models_{model}_{score_name}.png'),
                      figsize=(10, len(unique_metrics) * 0.5), palette=palette)


def draw_line_scores(ax, x_data, y_data, label, color="black", marker='o', markersize=5, linestyle='-', linewidth=1.5, display_text=True):
    
    sns.lineplot(
        x=x_data,
        y=y_data,
        ax=ax,
        color="black",
        marker=marker,
        markersize=markersize,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        legend=False
    )

    if display_text:
        for x, y in zip(x_data, y_data):
            if pd.notna(y):
                text = ax.text(x, y, f'{y:.2f}', color="black", ha='center', va='bottom')
                text.set_path_effects([withStroke(linewidth=3, foreground='white')])
    
        """
        x_numeric = pd.Series(range(len(x_data)))

        sns.regplot(
            x=x_numeric,
            y=pd.Series(y_data).astype(float),
            ax=ax,
            scatter=False,
            color=color,
            line_kws={"linestyle": "--", "linewidth": 0.5},
            label=f'{label} Trend',
        )
        """


def create_plots_conclusions_perMetric(output_directory, input_directory,promptfoo_runs, max_timeLine_length = TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
    all_data = []

    for timestamp, label in promptfoo_runs.items():
        file_name = f'output_cR_{timestamp}.json'
        file_path = os.path.join(input_directory, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            for item in data:
                all_data.append({
                    'TestRun': label,
                    'Model': item['metric']['model'],
                    'Metric': item['metric']['name'],
                    'Similarity Score': item['score']
                })

    df = pd.DataFrame(all_data)
    unique_metrics = sorted(df['Metric'].unique())
    unique_models = sorted(df['Model'].unique())
    test_runs = [label for label in promptfoo_runs.values()]

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_models))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for metric in unique_metrics:
        metric_data = df[df['Metric'] == metric]
        fig = plt.figure(figsize=(total_width, len(unique_models) * 2))

        gs = GridSpec(len(unique_models), 2, figure=fig, width_ratios=[1, max_timeLine_length])

        for i, model in enumerate(unique_models):
            model_data = metric_data[metric_data['Model'] == model]

            # Subplot for model names
            ax_model = fig.add_subplot(gs[i, 0])
            ax_model.text(0.5, 0.5, model, transform=ax_model.transAxes, va='center', ha='center', fontsize=12)
            ax_model.axis('off')

            # Subplot for plots
            ax_plot = fig.add_subplot(gs[i, 1])
            background_plot_alpha = 0.25
            background_plot_width = 0.8

            if plot_type == 'boxplot':
                sns.boxplot(
                    data=model_data,
                    x='TestRun',
                    y='Similarity Score',
                    color=palette[i],
                    ax=ax_plot,
                    width=background_plot_width,
                    boxprops=dict(alpha=background_plot_alpha),
                    whiskerprops=dict(alpha=background_plot_alpha),
                    capprops=dict(alpha=background_plot_alpha),
                    medianprops=dict(alpha=background_plot_alpha),
                    flierprops=dict(alpha=background_plot_alpha),
                    notch=True
                )
            elif plot_type == 'violinplot':
                sns.violinplot(
                    data=model_data,
                    x='TestRun',
                    y='Similarity Score',
                    ax=ax_plot,
                    color=palette[i],
                    width=background_plot_width,
                    alpha=background_plot_alpha,
                    linewidth=0
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            mean_scores = model_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            median_scores = model_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)

            draw_line_scores(ax_plot, test_runs, median_scores.values, 'Median', linestyle='--', linewidth=0.5, display_text=False, markersize=3)
            draw_line_scores(ax_plot, test_runs, mean_scores.values, 'Average', color=palette[i])

            ax_plot.set_ylabel('')
            ax_plot.set_xlabel('')
            ax_plot.set_yticks([-0.1, 0, 0.5, 1, 1.1])
            ax_plot.set_yticklabels(['', '0.0', '0.5', '1.0', ''])
            ax_plot.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax_plot.spines['top'].set_visible(False)
            ax_plot.spines['right'].set_visible(False)

            ax_plot.set_xticks(range(max_timeLine_length))

            if i == len(unique_models) - 1:
                xtick_labels = test_runs + [''] * (max_timeLine_length - len(test_runs))
                ax_plot.set_xticklabels(xtick_labels, rotation=45)
            else:
                ax_plot.set_xticklabels([])

        fig.suptitle(f'Similarity Scores - {metric}')
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        
        if file_type == 'png':
            save_path = os.path.join(output_directory, f'timeline_metrics_{metric}.png')
            plt.savefig(save_path, format='png', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")
        else:
            save_path = os.path.join(output_directory, f'timeline_metrics_{metric}.svg')
            plt.savefig(save_path, format='svg', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")

        plt.close()

def create_plots_conclusions_perMetric_singles(output_directory, input_directory, promptfoo_runs, max_timeLine_length = TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
    all_data = []

    for timestamp, label in promptfoo_runs.items():
        file_name = f'output_cR_{timestamp}.json'
        file_path = os.path.join(input_directory, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            for item in data:
                all_data.append({
                    'TestRun': label,
                    'Model': item['metric']['model'],
                    'Metric': item['metric']['name'],
                    'Similarity Score': item['score']
                })

    df = pd.DataFrame(all_data)
    unique_metrics = sorted(df['Metric'].unique())
    unique_models = sorted(df['Model'].unique())
    test_runs = [label for label in promptfoo_runs.values()]

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_models))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for metric in unique_metrics:
        metric_data = df[df['Metric'] == metric]

        for i, model in enumerate(unique_models):
            model_data = metric_data[metric_data['Model'] == model]
            fig = plt.figure(figsize=(total_width, 3.5))

            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, max_timeLine_length])

            # Subplot for model names
            ax_model = fig.add_subplot(gs[0, 0])
            ax_model.text(0.5, 0.5, model, transform=ax_model.transAxes, va='center', ha='center', fontsize=12)
            ax_model.axis('off')

            # Subplot for plots
            ax_plot = fig.add_subplot(gs[0, 1])
            background_plot_alpha = 0.25
            background_plot_width = 0.8

            if plot_type == 'boxplot':
                sns.boxplot(
                    data=model_data,
                    x='TestRun',
                    y='Similarity Score',
                    color=palette[i],
                    ax=ax_plot,
                    width=background_plot_width,
                    boxprops=dict(alpha=background_plot_alpha),
                    whiskerprops=dict(alpha=background_plot_alpha),
                    capprops=dict(alpha=background_plot_alpha),
                    medianprops=dict(alpha=background_plot_alpha),
                    flierprops=dict(alpha=background_plot_alpha)
                )
            elif plot_type == 'violinplot':
                sns.violinplot(
                    data=model_data,
                    x='TestRun',
                    y='Similarity Score',
                    ax=ax_plot,
                    color=palette[i],
                    width=background_plot_width,
                    alpha=background_plot_alpha,
                    linewidth=0
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            mean_scores = model_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            median_scores = model_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)

            draw_line_scores(ax_plot, test_runs, median_scores.values, 'Median', linestyle='--', linewidth=0.5, display_text=False, markersize=3)
            draw_line_scores(ax_plot, test_runs, mean_scores.values, 'Average', color=palette[i])

            ax_plot.set_ylabel('')
            ax_plot.set_xlabel('')
            ax_plot.set_yticks([-0.1, 0, 0.5, 1, 1.1])
            ax_plot.set_yticklabels(['', '0.0', '0.5', '1.0', ''])
            ax_plot.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax_plot.spines['top'].set_visible(False)
            ax_plot.spines['right'].set_visible(False)

            ax_plot.set_xticks(range(max_timeLine_length))

            
            xtick_labels = test_runs + [''] * (max_timeLine_length - len(test_runs))
            ax_plot.set_xticklabels(xtick_labels, rotation=45)
           

            fig.suptitle(f'Similarity Scores - {metric}')
            plt.tight_layout(rect=(0, 0, 1, 0.96))            

            if file_type == 'png':
                save_path = os.path.join(output_directory, f'timeline_metrics_{metric}_{model}.png')
                plt.savefig(save_path, format='png', dpi=600)
                print(f"Saving plot '{os.path.basename(save_path)}'")
            else:
                save_path = os.path.join(output_directory, f'timeline_metrics_{metric}_{model}.svg')
                plt.savefig(save_path, format='svg', dpi=600)
                print(f"Saving plot '{os.path.basename(save_path)}'")

            plt.close()


def create_plots_conclusions_perModel(output_directory, input_directory, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
    all_data = []

    for timestamp, label in promptfoo_runs.items():
        file_name = f'output_cR_{timestamp}.json'
        file_path = os.path.join(input_directory, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            for item in data:
                all_data.append({
                    'TestRun': label,
                    'Model': item['metric']['model'],
                    'Metric': item['metric']['name'],
                    'Similarity Score': item['score']
                })

    df = pd.DataFrame(all_data)
    unique_models = sorted(df['Model'].unique())
    unique_metrics = sorted(df['Metric'].unique())
    test_runs = [label for label in promptfoo_runs.values()]

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_metrics))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for model in unique_models:
        model_data = df[df['Model'] == model]
        fig = plt.figure(figsize=(total_width, len(unique_metrics) * 2))

        gs = GridSpec(len(unique_metrics), 2, figure=fig, width_ratios=[1, max_timeLine_length])

        for i, metric in enumerate(unique_metrics):
            metric_data = model_data[model_data['Metric'] == metric]

            # Subplot for metric names
            ax_metric = fig.add_subplot(gs[i, 0])
            ax_metric.text(0.5, 0.5, metric, transform=ax_metric.transAxes, va='center', ha='center', fontsize=12)
            ax_metric.axis('off')

            # Subplot for plots
            ax_plot = fig.add_subplot(gs[i, 1])
            background_plot_alpha = 0.25
            background_plot_width = 0.8

            if plot_type == 'boxplot':
                sns.boxplot(
                    data=metric_data,
                    x='TestRun',
                    y='Similarity Score',
                    color=palette[i],
                    ax=ax_plot,
                    width=background_plot_width,
                    boxprops=dict(alpha=background_plot_alpha),
                    whiskerprops=dict(alpha=background_plot_alpha),
                    capprops=dict(alpha=background_plot_alpha),
                    medianprops=dict(alpha=background_plot_alpha),
                    flierprops=dict(alpha=background_plot_alpha)
                )
            elif plot_type == 'violinplot':
                sns.violinplot(
                    data=metric_data,
                    x='TestRun',
                    y='Similarity Score',
                    ax=ax_plot,
                    color=palette[i],
                    width=background_plot_width,
                    alpha=background_plot_alpha,
                    linewidth=0
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            mean_scores = metric_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            median_scores = metric_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)

            draw_line_scores(ax_plot, test_runs, median_scores.values, 'Median', linestyle='--', linewidth=0.5, display_text=False, markersize=3)
            draw_line_scores(ax_plot, test_runs, mean_scores.values, 'Average', color=palette[i])

            ax_plot.set_ylabel('')
            ax_plot.set_xlabel('')
            ax_plot.set_yticks([-0.1, 0, 0.5, 1, 1.1])
            ax_plot.set_yticklabels(['', '0.0', '0.5', '1.0', ''])
            ax_plot.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax_plot.spines['top'].set_visible(False)
            ax_plot.spines['right'].set_visible(False)

            ax_plot.set_xticks(range(max_timeLine_length))

            if i == len(unique_metrics) - 1:
                xtick_labels = test_runs + [''] * (max_timeLine_length - len(test_runs))
                ax_plot.set_xticklabels(xtick_labels, rotation=45)
            else:
                ax_plot.set_xticklabels([])

        fig.suptitle(f'Similarity Scores - {model}')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        if file_type == 'png':
            save_path = os.path.join(output_directory, f'timeline_models_{model}.png')
            plt.savefig(save_path, format='png', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")
        else:
            save_path = os.path.join(output_directory, f'timeline_models_{model}.svg')
            plt.savefig(save_path, format='svg', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")

        plt.close()

def create_plots_conclusions_perModel_singles(output_directory, input_directory, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
    all_data = []

    for timestamp, label in promptfoo_runs.items():
        file_name = f'output_cR_{timestamp}.json'
        file_path = os.path.join(input_directory, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            for item in data:
                all_data.append({
                    'TestRun': label,
                    'Model': item['metric']['model'],
                    'Metric': item['metric']['name'],
                    'Similarity Score': item['score']
                })

    df = pd.DataFrame(all_data)
    unique_models = sorted(df['Model'].unique())
    unique_metrics = sorted(df['Metric'].unique())
    test_runs = [label for label in promptfoo_runs.values()]

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_metrics))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for model in unique_models:
        model_data = df[df['Model'] == model]

        for i, metric in enumerate(unique_metrics):
            metric_data = model_data[model_data['Metric'] == metric]
            fig = plt.figure(figsize=(total_width, 3.5))

            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, max_timeLine_length])

            # Subplot for metric names
            ax_metric = fig.add_subplot(gs[0, 0])
            ax_metric.text(0.5, 0.5, metric, transform=ax_metric.transAxes, va='center', ha='center', fontsize=12)
            ax_metric.axis('off')

            # Subplot for plots
            ax_plot = fig.add_subplot(gs[0, 1])
            background_plot_alpha = 0.25
            background_plot_width = 0.8

            if plot_type == 'boxplot':
                sns.boxplot(
                    data=metric_data,
                    x='TestRun',
                    y='Similarity Score',
                    color=palette[i],
                    ax=ax_plot,
                    width=background_plot_width,
                    boxprops=dict(alpha=background_plot_alpha),
                    whiskerprops=dict(alpha=background_plot_alpha),
                    capprops=dict(alpha=background_plot_alpha),
                    medianprops=dict(alpha=background_plot_alpha),
                    flierprops=dict(alpha=background_plot_alpha)
                )
            elif plot_type == 'violinplot':
                sns.violinplot(
                    data=metric_data,
                    x='TestRun',
                    y='Similarity Score',
                    ax=ax_plot,
                    color=palette[i],
                    width=background_plot_width,
                    alpha=background_plot_alpha,
                    linewidth=0
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            mean_scores = metric_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            median_scores = metric_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)

            draw_line_scores(ax_plot, test_runs, median_scores.values, 'Median', linestyle='--', linewidth=0.5, display_text=False, markersize=3)
            draw_line_scores(ax_plot, test_runs, mean_scores.values, 'Average', color=palette[i])

            ax_plot.set_ylabel('')
            ax_plot.set_xlabel('')
            ax_plot.set_yticks([-0.1, 0, 0.5, 1, 1.1])
            ax_plot.set_yticklabels(['', '0.0', '0.5', '1.0', ''])
            ax_plot.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax_plot.spines['top'].set_visible(False)
            ax_plot.spines['right'].set_visible(False)

            ax_plot.set_xticks(range(max_timeLine_length))

            xtick_labels = test_runs + [''] * (max_timeLine_length - len(test_runs))
            ax_plot.set_xticklabels(xtick_labels, rotation=45)

            fig.suptitle(f'Similarity Scores - {model}')
            plt.tight_layout(rect=(0, 0, 1, 0.96))

            if file_type == 'png':
                save_path = os.path.join(output_directory, f'timeline_models_{model}_{metric}.png')
                plt.savefig(save_path, format='png', dpi=600)
                print(f"Saving plot '{os.path.basename(save_path)}'")
            else:
                save_path = os.path.join(output_directory, f'timeline_models_{model}_{metric}.svg')
                plt.savefig(save_path, format='svg', dpi=600)
                print(f"Saving plot '{os.path.basename(save_path)}'")

            plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        main('20240701_151846')
    else:
        timestamp = sys.argv[1]
        main(timestamp)