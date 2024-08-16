import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke
import numpy as np

# Global variables for color interpolation
PLOT_COLOR_START = "#009B91"
PLOT_COLOR_END = "#275861"

promptfoo_runs = {
    "20240628_101349": "TestRun 2",
    "20240627_205930": "CompleteRun",
    "20240628_164154": "TestRun 3",
}

def create_interpolated_palette(start_color, end_color, n_colors):
    cmap = LinearSegmentedColormap.from_list("custom_palette", [start_color, end_color], N=n_colors)
    palette = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return palette

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

def create_timeLinePlots(promptfoo_runs, max_timeLine_length, file_type='png', PLOT_TYPE='violinplot'):
    all_data = []
    input_directory = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/promptfoo_data/processed/componentResults'

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
            background_plot_alpha = 0.2
            background_plot_width = 0.8

            if PLOT_TYPE == 'boxplot':
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
            elif PLOT_TYPE == 'violinplot':
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
                raise ValueError(f"Unknown plot type: {PLOT_TYPE}")

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

        output_directory = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/plots'
        
        if file_type == 'png':
            save_path = os.path.join(output_directory, f'timeline_metrics_{metric}.png')
            plt.savefig(save_path, format='png', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")
        else:
            save_path = os.path.join(output_directory, f'timeline_metrics_{metric}.svg')
            plt.savefig(save_path, format='svg', dpi=600)
            print(f"Saving plot '{os.path.basename(save_path)}'")

        plt.close()

def create_timeLinePlots_singles(promptfoo_runs, max_timeLine_length, file_type='png', PLOT_TYPE='violinplot'):
    all_data = []
    input_directory = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/promptfoo_data/processed/componentResults'

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
            background_plot_alpha = 0.2
            background_plot_width = 0.8

            if PLOT_TYPE == 'boxplot':
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
            elif PLOT_TYPE == 'violinplot':
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
                raise ValueError(f"Unknown plot type: {PLOT_TYPE}")

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

            output_directory = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/plots'
            

            if file_type == 'png':
                save_path = os.path.join(output_directory, f'timeline_metrics_{metric}_{model}.png')
                plt.savefig(save_path, format='png', dpi=600)
                print(f"Saving plot '{os.path.basename(save_path)}'")
            else:
                save_path = os.path.join(output_directory, f'timeline_metrics_{metric}_{model}.svg')
                plt.savefig(save_path, format='svg', dpi=600)
                print(f"Saving plot '{os.path.basename(save_path)}'")

            plt.close()


def create_timeLinePlots_perModel(promptfoo_runs, max_timeLine_length, file_type='png', PLOT_TYPE='violinplot'):
    all_data = []
    input_directory = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/promptfoo_data/processed/componentResults'

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

    for model in unique_models:
        model_data = df[df['Model'] == model]

        for i, metric in enumerate(unique_metrics):
            metric_data = model_data[model_data['Metric'] == metric]
            fig = plt.figure(figsize=(total_width, 3.5))

            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, max_timeLine_length])

            # Subplot for model names
            ax_model = fig.add_subplot(gs[0, 0])
            ax_model.text(0.5, 0.5, metric, transform=ax_model.transAxes, va='center', ha='center', fontsize=12)
            ax_model.axis('off')

            # Subplot for plots
            ax_plot = fig.add_subplot(gs[0, 1])
            background_plot_alpha = 0.2
            background_plot_width = 0.8

            if PLOT_TYPE == 'boxplot':
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
            elif PLOT_TYPE == 'violinplot':
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
                raise ValueError(f"Unknown plot type: {PLOT_TYPE}")
            
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

            output_directory = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/plots'

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

    create_timeLinePlots(promptfoo_runs, max_timeLine_length=8, file_type='png')
    create_timeLinePlots_singles(promptfoo_runs, max_timeLine_length=8, file_type='png')


"""

def create_plots_conclusions_perModel(output_directory, input_directory, plot_type='boxplot'):
    all_data = []
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_directory, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)

            for item in data:
                all_data.append({
                    'TestRun': os.path.splitext(file_name)[0],
                    'Model': item['metric']['model'],
                    'Metric': item['metric']['name'],
                    'Similarity Score': item['score']
                })

    df = pd.DataFrame(all_data)
    unique_metrics = sorted(df['Metric'].unique())
    unique_models = sorted(df['Model'].unique())
    test_runs = sorted(df['TestRun'].unique())

    sns.set_theme()
    palette = create_interpolated_palette(PLOT_COLOR_START, PLOT_COLOR_END, len(unique_metrics))

    for model in unique_models:
        model_data = df[df['Model'] == model]
        fig, axs = plt.subplots(len(unique_metrics), 1, sharex=True, figsize=(12, len(unique_metrics) * 2))

        fig.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.05, hspace=0.3)

        for i, metric in enumerate(unique_metrics):
            metric_data = model_data[model_data['Metric'] == metric]

            background_plot_alpha = 0.2
            background_plot_width = 0.3

            if plot_type == 'boxplot':
                sns.boxplot(
                    data=metric_data,
                    x='TestRun',
                    y='Similarity Score',
                    ax=axs[i],
                    color=palette[i],
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
                    ax=axs[i],
                    color=palette[i],
                    width=background_plot_width,
                    alpha=background_plot_alpha,
                    linewidth=0
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            mean_scores = metric_data.groupby('TestRun')['Similarity Score'].mean().reindex(test_runs)
            median_scores = metric_data.groupby('TestRun')['Similarity Score'].median().reindex(test_runs)

            draw_line_scores(axs[i], test_runs, median_scores.values, 'Median', linestyle='--', linewidth=1, display_text=False)
            draw_line_scores(axs[i], test_runs, mean_scores.values, 'Average', color=palette[i])

            axs[i].set_ylabel('')
            axs[i].set_yticks([0, 1])
            axs[i].set_yticklabels(['0', '1'])
            axs[i].grid(True, axis='y', linestyle='--', linewidth=0.5)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)

            axs[i].text(-0.1, 0.5, metric, transform=axs[i].transAxes, va='center', ha='right', fontsize=12)
            
        axs[-1].set_xticks(range(len(test_runs)))
        axs[-1].set_xticklabels(test_runs, rotation=45)
        fig.suptitle(f'Similarity Scores - {model}')
        #plt.legend(loc='upper right')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        save_path = os.path.join(output_directory, f'conclusions_models_{model}.png')
        plt.savefig(save_path)
        plt.close()


"""