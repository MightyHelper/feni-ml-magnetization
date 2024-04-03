from pathlib import Path

import matplotlib
import numpy as np
import typer
import pandas as pd
from matplotlib import pyplot as plt

from lammps.nanoparticle import Nanoparticle
from lammps.nanoparticle_locator import NanoparticleLocator
from utils import NanoparticleName

DATASET_VERSIONS_ROOT = Path("../dataset_versions")

plot = typer.Typer(add_completion=False, no_args_is_help=True)
dataset = typer.Typer(add_completion=False, no_args_is_help=True)


def get_default_dataset():
    return sorted(DATASET_VERSIONS_ROOT.glob("*.csv"), key=lambda x: int(x.stem.split("_")[0]))[-1]


# @dataset.command()
# def type_distribution(path: Path = typer.Argument(None, help="Path to the dataset")):
#     if path is None: path = get_default_dataset()
#     dataset_csv = pd.read_csv(path)
#     nanoparticle_names = dataset_csv['name'].apply(lambda x: NanoparticleName.parse(x))
#     # csv['name'].apply(lambda x: NanoparticleName.parse(x).shape)
#
#     plot_df = pd.DataFrame({
#         'type': nanoparticle_names.apply(lambda x: x.shape),
#         'distribution': nanoparticle_names.apply(lambda x: x.distribution)
#     })
#
#     plot_df = plot_df.groupby(['type', 'distribution']).size().unstack().fillna(0)
#     plot_df.plot(kind='bar', stacked=True)
#     plt.tight_layout()
#     plt.show()
#

@dataset.command()
def type_distribution_bar(path: Path = typer.Argument(None, help="Path to the dataset"), by: str = "type"):
    plot_df = get_distribution_df(path)
    if by == "type":
        gb_col = 'distribution_type'
        ngb_col = 'type'
    elif by == 'distribution_type':
        gb_col = 'distribution_type'
        ngb_col = 'type'
    else:
        raise ValueError(f"Invalid value for 'by' argument: {by}")

    unique_types = plot_df[gb_col].unique()
    num_types = len(unique_types)

    num_cols = 5  # Number of subplots per row
    num_rows = -(-num_types // num_cols)  # Ceiling division to calculate number of rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True)
    if num_rows == 1:
        axs = axs.reshape(1, -1)

    for i, nanoparticle_type in enumerate(unique_types):
        row = i // num_cols
        col = i % num_cols
        subset_df = plot_df[plot_df[gb_col] == nanoparticle_type]
        subset_df.groupby(ngb_col).size().plot(kind='bar', stacked=True, ax=axs[row, col])
        axs[row, col].set_title(f"Type: {nanoparticle_type}")
        axs[row, col].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


@dataset.command()
def type_distribution_heatmap(path: Path = typer.Argument(None, help="Path to the dataset"), log: bool = False):
    plot_df = get_distribution_df(path)
    og_df = plot_df.copy()
    if log:
        # plot_df = plot_df.applymap(lambda x: np.log(x) if x > 0 else 0)
        plt.title("Logarithmic heatmap of nanoparticle distribution types")
    else:
        plt.title("Heatmap of nanoparticle distribution types")
    plt.imshow(plot_df, cmap='gray', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=plot_df.min(), vmax=plot_df.max()))
    plt.xticks(range(len(plot_df.columns)), plot_df.columns, rotation=90)
    plt.yticks(range(len(plot_df.index)), plot_df.index)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def get_distribution_df(path):
    if path is None: path = get_default_dataset()
    dataset_csv = pd.read_csv(path)
    nanoparticle_names = dataset_csv['name'].apply(lambda x: NanoparticleName.parse(x))
    plot_df = pd.DataFrame({
        'type': nanoparticle_names.apply(lambda x: x.shape),
        'distribution_type': nanoparticle_names.apply(lambda x: x.distribution)
    })
    plot_df = plot_df.groupby(['type', 'distribution_type']).size().unstack().fillna(0)
    return plot_df


plot.add_typer(dataset, name="dataset", help="Plot dataset")
