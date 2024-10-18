import multiprocessing
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import rich.progress
import typer
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from networkx.drawing.nx_pydot import graphviz_layout
from rich import print as rprint
from rich.progress import Progress

import config.config
from lammps.nanoparticle import Nanoparticle
from utils import NanoparticleName

DATASET_VERSIONS_ROOT = Path("../dataset_versions")

plot = typer.Typer(add_completion=False, no_args_is_help=True)
dataset = typer.Typer(add_completion=False, no_args_is_help=True)
executions = typer.Typer(add_completion=False, no_args_is_help=True)


def get_default_dataset():
    return sorted(DATASET_VERSIONS_ROOT.glob("*.csv"), key=lambda x: int(x.stem.split("_")[0]))[-1]


@dataset.command()
def type_distribution_bar(path: Path = typer.Argument(None, help="Path to the dataset"), by: str = "type"):
    plot_df = get_distribution_df(path)[0]
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
def type_distribution_heatmap(
      path: Path = typer.Argument(None, help="Path to the dataset"),
      log: bool = False,
      title: bool = True
):
    plot_df = get_distribution_df(path)[0]
    if log:
        plot_df = plot_df.applymap(lambda x: np.log(x) if x > 0 else 0)
        if title: plt.title("Logarithmic heatmap of nanoparticle distribution types")
    else:
        if title: plt.title("Heatmap of nanoparticle distribution types")

    cmap = get_cmap()
    # Invert color map
    plt.imshow(
        plot_df,
        cmap=cmap,
        interpolation='nearest'
    )
    plt.xticks(range(len(plot_df.columns)), plot_df.columns, rotation=90)
    plt.yticks(range(len(plot_df.index)), plot_df.index)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def get_cmap():
    # Define the colors and positions
    colors = [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 1)]  # Red, Orange, Green
    positions = [0, 0.01, 0.1, 1]  # More detail between 0 and 0.1
    # Create the colormap
    cmap_name = 'red_orange_green'
    cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(positions, colors)))
    return cmap





class TrieNode:
    def __init__(self, key: str):
        self.children: dict[str, TrieNode] = {}
        self.key = key
        self.count = 0
    def __getitem__(self, item: str) -> Optional['TrieNode']:
        if len(item) == 0:
            return self
        for char, child in self.children.items():
            if item.startswith(char):
                return child[item[len(char):]]
        return None

    def add(self, label: str):
        self.count += 1
        if len(label) == 0:
            return
        char = label[0]
        if char not in self.children:
            self.children[char] = TrieNode(key=char)
        self.children[char].add(label[1:])


    def shake(self) -> tuple[str, 'TrieNode']:
        if len(self.children) == 1:
            char, child = list(self.children.items())[0]
            new_char, new_child = child.shake()
            return self.key + new_char, new_child
        children = list(self.children.items())
        for char, child in children:
            new_char, new_child = child.shake()
            del self.children[char]
            self.children[new_char] = new_child
        return self.key, self

    def __repr__(self):
        return repr(self.children)

    def max_count(self) -> int:
        return max([child.max_count() for child in self.children.values()], default=self.count)


class Trie:
    def __init__(self):
        self.root = TrieNode("")
    def insert(self, word):
        self.root.add(word)

    def build_from_list(self, words):
        for word in words:
            self.insert(word)

    def shake(self):
        self.root = self.root.shake()[1]

    def __repr__(self):
        return "Trie(" + repr(self.root) + ")"

    def max_count(self):
        return self.root.max_count()

    def __getitem__(self, item):
        return self.root[item]
import matplotlib.pyplot as plt


import networkx as nx
def add_node(graph: nx.DiGraph, label: str):
    print("+ ", label)
    graph.add_node(label)


def add_edge(graph: nx.DiGraph, parent: str, child: str):
    print("+ E ", parent, child)
    graph.add_edge(parent, child)


def add_edges(trie: Trie, node: TrieNode, graph: nx.DiGraph, parent: str | None = None, label: str = ''):
    if parent is None:
        add_node(graph, label)
        add_edges(trie, node, graph, label, label)
        return
    for char, child in node.children.items():
        add_node(graph, label + char)
        add_edge(graph, parent, label + char)
        add_edges(trie, child, graph, label + char, label + char)

def serialize_edges(trie: Trie, node: TrieNode, parent: str | None = None, label: str = ''):
    ser_node = lambda label: f"N '{label}' {trie[label].count}"
    ser_edge = lambda parent, child: f"E '{parent}', '{child}'"
    if parent is None:
        print(ser_node(label))
        serialize_edges(trie, node, label, label)
        return
    for char, child in node.children.items():
        print(ser_node(label + char))
        print(ser_edge(parent, label + char))
        serialize_edges(trie, child, label + char, label + char)


def serialize_trie(trie: Trie):
    serialize_edges(trie, trie.root)


@dataset.command()
def type_distribution_trie(
      path: Path = typer.Argument(None, help="Path to the dataset"),
):
    nanoparticle_names = get_distribution_df(path)[1]
    trie = Trie()
    unique = nanoparticle_names.apply(lambda x: x.distribution)
    trie.build_from_list(unique)
    trie.shake()
    # plot_trie(trie)
    serialize_trie(trie)


def plot_trie(trie: Trie):
    G = nx.DiGraph()
    add_edges(trie, trie.root, G)

    # Create a clone of the graph, but rename the nodes with their index
    G2 = nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes)})

    # pos = nx.bfs_layout(G, start="")

    pos = graphviz_layout(G2, prog="dot")
    # Map indices back to the original node names
    pos = {list(G.nodes)[node]: i for node, i in pos.items()}
    # Rotate 90 degrees
    pos = {node: (-y, x * 2) for node, (x, y) in pos.items()}
    plt.figure(figsize=(10, 10))  # Set the figure size
    # Colormap
    cmap = get_cmap()
    # Scale the color on the map based on the count on the trie node
    max_count = trie.max_count()
    trie_nodes = [trie[node] for node in G.nodes]
    print([(trie_node.key if trie_node is not None else "?") for trie_node in trie_nodes])
    colors = [cmap(trie_node.count / max_count) for trie_node in trie_nodes]
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=450,
        node_color=colors,
        node_shape="s",
        font_weight="bold",
        font_size=10,
        arrows=True,
        font_color="white",
        bbox=dict(facecolor=(0.5, 0.5, 0.5, 0.5), edgecolor='black', boxstyle='round,pad=0.2')
    )
    plt.title("Trie Visualization")
    print(max_count)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_count)), label="Count", orientation="horizontal", shrink=0.5)
    # set background color
    plt.gca().set_facecolor((0.75, 0.75, 0.75))
    plt.tight_layout()
    plt.show()


def convert_distribution(name: NanoparticleName) -> str:
    if name.distribution_type == "Multilayer" and name.distribution.startswith("Multilayer.2."):
        if name.interface == "Normal":
            return "Janus"
        else:
            return "Janus-R"
    return name.distribution_type


def get_distribution_df(path):
    if path is None: path = get_default_dataset()
    dataset_csv = pd.read_csv(path)
    nanoparticle_names = dataset_csv['name'].apply(lambda x: NanoparticleName.parse(x))
    plot_df = pd.DataFrame({
        'type': nanoparticle_names.apply(lambda x: x.shape),
        'distribution_type': nanoparticle_names.apply(convert_distribution)
    })
    return plot_df.groupby(['type', 'distribution_type']).size().unstack().fillna(0), nanoparticle_names


@executions.command()
def nano(path: Path = typer.Argument(..., help="Path to the execution")):
    if not path.exists():
        raise FileNotFoundError(f"Execution path {path} does not exist!")

    nanoparticle = Nanoparticle.from_executed(path)
    nanoparticle.plot()


def get_execution_count() -> int:
    return len(list(config.config.LOCAL_EXECUTION_PATH.iterdir()))


def get_execution_paths() -> Generator[Path, None, None]:
    for execution in config.config.LOCAL_EXECUTION_PATH.iterdir():
        if "simulation" in execution.name:
            yield execution


def get_executions() -> Generator[Nanoparticle, None, None]:
    for execution in get_execution_paths():
        yield Nanoparticle.from_executed(execution)


def is_weak(path: Path) -> Nanoparticle | None:
    nanoparticle = Nanoparticle.from_executed(path)
    if nanoparticle.is_weak():
        return nanoparticle
    return None


@executions.command()
def weak():
    nanoparticles = []
    # with Progress() as progress:
    #     task = progress.add_task("Getting weak nanoparticles", total=get_execution_count())
    # for nanoparticle in get_executions():
    #     if nanoparticle.is_weak():
    #         nanoparticles.append(nanoparticle)
    #     progress.update(task, advance=1)
    paths = list(get_execution_paths())
    with multiprocessing.Pool() as pool:
        nanoparticles = [n for n in rich.progress.track(pool.imap_unordered(is_weak, paths), total=len(paths), transient=True, description="Loading nanoparticles") if n is not None]
        rprint(list(nanoparticles))


plot.add_typer(dataset, name="dataset", help="Plot dataset")
plot.add_typer(executions, name="executions", help="Plot executions")
