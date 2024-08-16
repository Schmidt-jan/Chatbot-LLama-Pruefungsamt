from calendar import c
import os
import json
import sys
from turtle import color
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import norm  # Sicherstellen, dass norm korrekt importiert wird
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patheffects import withStroke
#from scipy.stats import linregress
from confidenceinterval.bootstrap import bootstrap_ci

HTWG_COLOR_1 = "#009b91"

# Global variables for color interpolation
PLOT_COLOR_START = "#009b91"#009B91
PLOT_COLOR_END = "#009b91" #275861

# Maximum number of test runs to display in timeline plots
TIMELINE_MAX_LENGTH = 6

promptfoo_runs = {
    #"20240628_101349": "TestRun 2",
    "20240627_205930": "CompleteRun",
    #"20240628_164154": "TestRun 3",
    #"20240629_224248": "BenniRun"
    #"20240630_113545": "BenniRunFinetune1",
    #"20240630_220713": "BenniRunFinetune2",
    #"20240701_151846": "BenniRunFinetune3",
    "20240801_123555": "Finetune-QAC-Pairs",
    #"20240729_234749": "OpenAI",
    "20240805_231203": "Finetune-QAC-COT"
}

PLOT_TYPE = 'boxplot'    # 'boxplot' or 'violinplot'
FILE_TYPE = 'png'        # 'png' or 'svg'

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

    
    create_plots_conclusions_perMetric(conclusions_metrics_combined_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    print("")
    create_plots_conclusions_perMetric_ci(conclusions_metrics_combined_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    print("")
    create_plots_conclusions_perModel(conclusions_models_combined_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    print("")
    create_plots_conclusions_perModel_ci(conclusions_models_combined_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    print("")
    create_plots_conclusions_perMetric_singles(conclusions_metrics_singles_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    print("")
    create_plots_conclusions_perModel_singles(conclusions_models_singles_path, component_results_path, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type=FILE_TYPE, plot_type=PLOT_TYPE)
    print("")
    generate_bar_plots_perMetric(singles_metrics_path, input_file_cR)
    print("")
    generate_bar_plots_perModel(singles_models_path, input_file_cR)


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

def compute_bootstrap_ci(data, confidence=0.95, n_resamples=9999):
    """
    Berechnet das Bootstrap-Konfidenzintervall für eine gegebene Datenreihe.
    
    :param data: Eine Liste oder ein Array von numerischen Werten.
    :param confidence: Das Konfidenzniveau (default ist 95%).
    :param n_resamples: Die Anzahl der Bootstrap-Stichproben (default ist 9999).
    :return: Ein Tuple (lower_bound, upper_bound).
    """
    # Dummy-Werte für y_true, da wir keinen tatsächlichen Vergleich haben
    y_true = np.zeros_like(data)
    
    # Berechnung des Konfidenzintervalls
    lower_bound, upper_bound = bootstrap_ci(
        y_true=y_true,
        y_pred=data,
        metric=np.average,
        confidence_level=confidence,
        n_resamples=n_resamples
    )
    return lower_bound, upper_bound

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


def create_plots_conclusions_perMetric_2(output_directory, input_directory, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
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

    # Erstellen eines vollständigen DataFrames mit NaN für fehlende Werte
    df_complete = pd.DataFrame(
        [(metric, model, run) for metric in unique_metrics for model in unique_models for run in test_runs],
        columns=['Metric', 'Model', 'TestRun']
    ).merge(df, on=['Metric', 'Model', 'TestRun'], how='left')

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_models))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for metric in unique_metrics:
        metric_data = df_complete[df_complete['Metric'] == metric]
        fig = plt.figure(figsize=(total_width, len(unique_models) * 2))

        gs = GridSpec(len(unique_models), 2, figure=fig, width_ratios=[1, max_timeLine_length])

        for i, model in enumerate(unique_models):
            model_data = metric_data[metric_data['Model'] == model]

            # Subplot für Modellnamen
            ax_model = fig.add_subplot(gs[i, 0])
            ax_model.text(0.5, 0.5, model, transform=ax_model.transAxes, va='center', ha='center', fontsize=12)
            ax_model.axis('off')

            # Subplot für Plots
            ax_plot = fig.add_subplot(gs[i, 1])
            ax_plot.set_xlim(0, max_timeLine_length)
            ax_plot.set_xticks(range(max_timeLine_length))
            ax_plot.set_xticklabels([''] * max_timeLine_length)
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
                    notch=False
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

def create_plots_conclusions_perMetric(output_directory, input_directory, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
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

    # Erstellen eines vollständigen DataFrames mit NaN für fehlende Werte
    df_complete = pd.DataFrame(
        [(metric, model, run) for metric in unique_metrics for model in unique_models for run in test_runs],
        columns=['Metric', 'Model', 'TestRun']
    ).merge(df, on=['Metric', 'Model', 'TestRun'], how='left')

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_models))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for metric in unique_metrics:
        metric_data = df_complete[df_complete['Metric'] == metric]
        fig = plt.figure(figsize=(total_width, len(unique_models) * 2))

        gs = GridSpec(len(unique_models), 2, figure=fig, width_ratios=[1, max_timeLine_length])

        for i, model in enumerate(unique_models):
            model_data = metric_data[metric_data['Model'] == model]

            # Subplot für Modellnamen
            ax_model = fig.add_subplot(gs[i, 0])
            ax_model.text(0.5, 0.5, model, transform=ax_model.transAxes, va='center', ha='center', fontsize=12)
            ax_model.axis('off')

            # Subplot für Plots
            ax_plot = fig.add_subplot(gs[i, 1])
            ax_plot.set_xlim(0, max_timeLine_length)
            ax_plot.set_xticks(range(max_timeLine_length))
            ax_plot.set_xticklabels([''] * max_timeLine_length)
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
                    notch=False
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
            ax_plot.set_xlim(0, max_timeLine_length)
            ax_plot.set_xticks(range(max_timeLine_length))
            ax_plot.set_xticklabels([''] * max_timeLine_length)
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

def create_plots_conclusions_perModel_singles_2(output_directory, input_directory, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
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

    # Erstellen eines vollständigen DataFrames mit NaN für fehlende Werte
    df_complete = pd.DataFrame(
        [(model, metric, run) for model in unique_models for metric in unique_metrics for run in test_runs],
        columns=['Model', 'Metric', 'TestRun']
    ).merge(df, on=['Model', 'Metric', 'TestRun'], how='left')

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_metrics))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for model in unique_models:
        model_data = df_complete[df_complete['Model'] == model]

        for i, metric in enumerate(unique_metrics):
            metric_data = model_data[model_data['Metric'] == metric]
            fig = plt.figure(figsize=(total_width, 3.5))

            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, max_timeLine_length])

            # Subplot für Metriknamen
            ax_metric = fig.add_subplot(gs[0, 0])
            ax_metric.text(0.5, 0.5, metric, transform=ax_metric.transAxes, va='center', ha='center', fontsize=12)
            ax_metric.axis('off')

            # Subplot für Plots
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

            # xtick_labels wird so gesetzt, dass es alle TestRuns enthält und mit Spacern auffüllt
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



def draw_line_scores_ci(ax, xticks, ydata, label, color='black', linestyle='-', linewidth=1, display_text=True, markersize=5):
    ax.plot(xticks, ydata, color=color, linestyle=linestyle, linewidth=linewidth, marker='o', markersize=markersize, label=label)
    if display_text:
        for x, y in zip(xticks, ydata):
            ax.text(x, y, f'{y:.2f}', color=color, ha='center', va='bottom', fontsize=8)

def calculate_mean_ci(data, confidence=0.95):
    """Calculate confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    h = sem * norm.ppf((1 + confidence) / 2)
    return mean - h, mean + h

def bootstrap_median_ci(data, n_bootstrap=1000, confidence=0.95):
    """Calculate confidence interval for the median using bootstrap."""
    medians = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower = np.percentile(medians, lower_percentile)
    upper = np.percentile(medians, upper_percentile)
    
    return lower, upper

def create_plots_conclusions_perMetric_ci(output_directory, input_directory, promptfoo_runs, max_timeLine_length=10, file_type='png', plot_type='violinplot'):
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

    # Erstellen eines vollständigen DataFrames mit NaN für fehlende Werte
    df_complete = pd.DataFrame(
        [(metric, model, run) for metric in unique_metrics for model in unique_models for run in test_runs],
        columns=['Metric', 'Model', 'TestRun']
    ).merge(df, on=['Metric', 'Model', 'TestRun'], how='left')

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_models))

    fixed_width_per_testrun = 2
    total_width = fixed_width_per_testrun * max_timeLine_length

    for metric in unique_metrics:
        metric_data = df_complete[df_complete['Metric'] == metric]
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
            ax_plot.set_xlim(0, max_timeLine_length)
            ax_plot.set_xticks(range(max_timeLine_length))
            ax_plot.set_xticklabels([''] * max_timeLine_length)
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
                    notch=False
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

            # Calculate mean and confidence intervals
            mean_scores = model_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            mean_ci = [calculate_mean_ci(model_data[model_data['TestRun'] == run]['Similarity Score']) for run in test_runs]
            lower_mean_ci, upper_mean_ci = zip(*mean_ci)
            lower_mean_ci = np.array(lower_mean_ci)
            upper_mean_ci = np.array(upper_mean_ci)

            # Plot mean scores with confidence intervals
            draw_line_scores_ci(ax_plot, test_runs, mean_scores.values, 'Mean', color="black")
            ax_plot.fill_between(test_runs, lower_mean_ci, upper_mean_ci, color='black', alpha=0.2, label='Mean CI')

            # Median scores and CI using bootstrap
            median_scores = model_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)
            median_ci = [bootstrap_median_ci(model_data[model_data['TestRun'] == run]['Similarity Score']) for run in test_runs]
            lower_median_ci, upper_median_ci = zip(*median_ci)
            lower_median_ci = np.array(lower_median_ci)
            upper_median_ci = np.array(upper_median_ci)

            # Plot median scores with confidence intervals
            draw_line_scores_ci(ax_plot, test_runs, median_scores.values, 'Median', color='darkgrey', linestyle='--')
            ax_plot.fill_between(test_runs, lower_median_ci, upper_median_ci, color='darkgrey', alpha=0.2, label='Median CI')

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
            save_path = os.path.join(output_directory, f'timeline_metrics_{metric}_ci.png')
            plt.savefig(save_path, format='png', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")
        else:
            save_path = os.path.join(output_directory, f'timeline_metrics_{metric}_ci.svg')
            plt.savefig(save_path, format='svg', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")

        plt.close()


def create_plots_conclusions_perModel_ci(output_directory, input_directory, promptfoo_runs, max_timeLine_length=TIMELINE_MAX_LENGTH, file_type='png', plot_type='violinplot'):
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
            ax_plot.set_xlim(0, max_timeLine_length)
            ax_plot.set_xticks(range(max_timeLine_length))
            ax_plot.set_xticklabels([''] * max_timeLine_length)
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
                    flierprops=dict(alpha=background_plot_alpha),
                    notch=False
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
            
            # Calculate mean and confidence intervals
            mean_scores = metric_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            mean_ci = [calculate_mean_ci(metric_data[metric_data['TestRun'] == run]['Similarity Score']) for run in test_runs]
            lower_mean_ci, upper_mean_ci = zip(*mean_ci)
            lower_mean_ci = np.array(lower_mean_ci)
            upper_mean_ci = np.array(upper_mean_ci)

            # Plot mean scores with confidence intervals
            draw_line_scores_ci(ax_plot, test_runs, mean_scores.values, 'Mean', color="black")
            ax_plot.fill_between(test_runs, lower_mean_ci, upper_mean_ci, color='black', alpha=0.2, label='Mean CI')

            # Median scores and CI using bootstrap
            median_scores = metric_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)
            median_ci = [bootstrap_median_ci(metric_data[metric_data['TestRun'] == run]['Similarity Score']) for run in test_runs]
            lower_median_ci, upper_median_ci = zip(*median_ci)
            lower_median_ci = np.array(lower_median_ci)
            upper_median_ci = np.array(upper_median_ci)

            # Plot median scores with confidence intervals
            # draw_line_scores_ci(ax_plot, test_runs, median_scores.values, 'Median', color='darkgrey', linestyle='--')
            # ax_plot.fill_between(test_runs, lower_median_ci, upper_median_ci, color='darkgrey', alpha=0.2, label='Median CI')

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
            save_path = os.path.join(output_directory, f'timeline_models_{model}_ci.png')
            plt.savefig(save_path, format='png', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")
        else:
            save_path = os.path.join(output_directory, f'timeline_models_{model}_ci.svg')
            plt.savefig(save_path, format='svg', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")

        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        main('00')
    else:
        timestamp = sys.argv[1]
        main(timestamp)