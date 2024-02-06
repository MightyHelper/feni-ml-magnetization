import logging
import os

import pandas as pd
from matplotlib import pyplot as plt
from rich.highlighter import RegexHighlighter
from rich.progress import Progress, TaskID

from cli_parts.number_highlighter import h
from config import config
from model.live_execution import LiveExecution


def confirm(message):
    res = input(f"{message} (y/n) ")
    if res != "y":
        raise ValueError("Oki :c")


def do_plots(df: pd.DataFrame, by: str = "Shape", field: str = "ratio_ni", min_acceptable: float | None = None,
             max_acceptable: float | None = None, source: str | None = None):
    multi_plots(
        df,
        source,
        (by, field, min_acceptable, max_acceptable),
    )


def multi_plots(
        df: pd.DataFrame,
        source: str | None = None,
        *plot_args: tuple[str, str, float | None, float | None]
) -> plt.Figure:
    # Create a figure with a 1x2 grid (1 row, 2 columns)
    fig, all_axes = plt.subplots(nrows=len(plot_args), ncols=2, figsize=(15, 4 * len(plot_args)))
    if len(plot_args) == 1:
        all_axes = [all_axes]

    for plt_idx, plt_args in enumerate(plot_args):
        axes = all_axes[plt_idx]
        by: str = plt_args[0]
        field: str = plt_args[1]
        min_acceptable: float | None = plt_args[2]
        max_acceptable: float | None = plt_args[3]

        # print(df.to_string())
        # Get unique shapes from the DataFrame
        unique_shapes = df[by].unique()

        # Subplot 1: Stacked Histogram
        hist_data = [df[df[by] == shape][field] for shape in unique_shapes]
        axes[0].hist(hist_data, bins=max(len(df) // 20, 5), stacked=True, label=unique_shapes)
        axes[0].set_title(f'Distribution of {field} by {by}')
        axes[0].set_xlabel(f'{field} (Value)')
        axes[0].set_ylabel('Count')
        if min_acceptable is not None:
            axes[0].axvline(min_acceptable, color="red", label="Min acceptable")
        if max_acceptable is not None:
            axes[0].axvline(max_acceptable, color="green", label="Max acceptable")
        axes[0].legend()

        # Subplot 2: Boxplot
        df.boxplot(column=field, by=by, ax=axes[1])
        axes[1].set_title(f'Boxplot of {field} by {by}')
        axes[1].set_xlabel(by)
        axes[1].set_ylabel(f'{field} (Value)')
        if min_acceptable is not None:
            axes[1].axhline(min_acceptable, color="red", label="Min acceptable")
        if max_acceptable is not None:
            axes[1].axhline(max_acceptable, color="green", label="Max acceptable")

    # Adjust layout to prevent overlap
    # Set title
    source = f"({source})" if source is not None else "nanoparticles"
    fig.suptitle(f"Distribution plots for {source}")
    fig.tight_layout()
    return fig


def scatter(df: pd.DataFrame, by: str, x: str, y: str):
    # Create a figure with a 1x2 grid (1 row, 2 columns)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

    # Get unique shapes from the DataFrame
    unique_shapes = df[by].unique()

    # Subplot 1: Stacked Histogram
    for shape in unique_shapes:
        axes.scatter(df[df[by] == shape][x], df[df[by] == shape][y], label=shape)
    axes.set_title(f'Scatter plot of {x} and {y} by {by}')
    axes.set_xlabel(f'{x} (Value)')
    axes.set_ylabel(f'{y} (Value)')

    # Adjust layout to prevent overlap
    # Set title
    fig.tight_layout()
    # Add legend
    axes.legend()
    return fig

def correct_highlighter(column: str, value) -> str:
    if value in ["True", "False"]:
        return render_boolean(value == "True")
    desired_by_column = {
        "ratio_fe": config.DESIRED_FE_RATIO,
        "ratio_ni": config.DESIRED_NI_RATIO,
        "total": config.DESIRED_ATOM_COUNT,
        "fe": config.DESIRED_FE_ATOM_COUNT,
        "ni": config.DESIRED_NI_ATOM_COUNT
    }
    limited_values = ['ratio_ni', 'total']
    if column in desired_by_column:
        val = float(value)
        val_txt = f"{val:.4f}" if 'ratio' in column else f"{val:.0f}"
        target = desired_by_column[column]
        acceptable_variance = config.DESIRED_MAX_RATIO_VARIANCE if target < 2 else config.DESIRED_MAX_ATOM_COUNT_VARIANCE
        if column in limited_values and abs(val - target) > acceptable_variance:
            return f"[b red u blink]{val_txt}[/b red u blink]"
        acceptable_variance /= 2
        x = 1 - min(abs(val - target) / acceptable_variance, 1)
        color = lerp_green_red(x)
        return f"[{color}]{val_txt}[/{color}]"
    else:
        return h(value)


class ZeroHighlighter(RegexHighlighter):
    """Apply style to anything that looks like non-zero."""
    base_style = "zero."
    highlights = [
        r"(^(?P<zero>0+(.0+)))|([^.\d](?P<zero_1>0+(.0+))$)|(^(?P<zero_2>0+(.0+))$)|([^.\d](?P<zero_3>0+(.0+))[^.\d])"]


def render_boolean(value: bool) -> str:
    # Render using emotes
    return "[green]✓[/green]" if value else "[red]✗[/red]"


def lerp_green_red(value: float) -> str:
    # Return hex of color
    green = int(value * 255)
    red = int((1 - value) * 255)
    green_hex = hex(green)[2:]
    red_hex = hex(red)[2:]
    return f"#{red_hex:0>2}{green_hex:0>2}00".upper()


def add_task(execution: LiveExecution, progress: Progress, tasks: dict[str, TaskID]) -> None:
    logging.info(f"Found running execution: {execution}")
    tasks[str(execution.folder)] = progress.add_task(
        f"{os.path.basename(execution.folder)} ({execution.title})",
        total=execution.get_total_execution_length()
    )

def remove_old_tasks(progress: Progress, running: list[LiveExecution], tasks: dict[str, TaskID]):
    keys_to_remove = []
    for folder in tasks.keys():
        if folder not in [execution.folder for execution in running]:
            logging.info(f"Execution {folder} has finished")
            try:
                progress.remove_task(tasks[folder])
                keys_to_remove.append(folder)
            except KeyError:
                pass
    for key in keys_to_remove:
        del tasks[key]


def add_new_tasks(progress: Progress, running: list[LiveExecution], tasks: dict[str, TaskID]):
    for execution in running:
        if execution.folder not in tasks:
            add_task(execution, progress, tasks)


def update_tasks(progress: Progress, running: list[LiveExecution], tasks: dict[str, TaskID]):
    for execution in running:
        progress.update(
            tasks[str(execution.folder)],
            completed=execution.step,
            total=execution.get_total_execution_length()
        )
    progress.refresh()


def create_tasks(progress: Progress, running: list[LiveExecution], tasks: dict[str, TaskID]):
    for execution in running:
        add_task(execution, progress, tasks)
