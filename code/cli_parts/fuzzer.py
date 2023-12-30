import logging
import re
from pathlib import Path
from typing import Callable, Annotated

from bayes_opt import BayesianOptimization
import typer
from rich import print as rprint

import config
import poorly_coded_parser as parser
import utils
from service import executor_service

fuzzer = typer.Typer(add_completion=False, no_args_is_help=True, name="fuzzer")


def get_full_function(path: str) -> tuple[list, Callable]:
    with open(path, 'r') as f:
        contents = f.read()
    keys = re.findall(r"{{(.*?)}}", contents)

    def fuzz(**kwargs):
        kwargs = {k: str(v) for k, v in kwargs.items()}
        logging.debug(kwargs)
        parsed_path, nano_builder = parser.PoorlyCodedParser.parse_single_shape(path, False, replacements=kwargs)
        nanoparticle = nano_builder.build(title='Fuzzing ' + nano_builder.title)
        executed_path, executed_nano = executor_service.execute_single_nanoparticle((
            parsed_path, nanoparticle),
            at="local",
            test=True
        )
        result = executed_nano.asdict()
        atom_count = result['total']
        ratio = result['ratio_ni']
        return atom_count, ratio, nanoparticle

    return keys, fuzz


@fuzzer.command()
def bayes(
        path: Annotated[Path, typer.Argument(help="The path to the nanoparticle (the [green].in[/green] file)")],
        target_atom_count: Annotated[int, typer.Option(help="The target atom count")] = config.DESIRED_ATOM_COUNT,
        target_ratio: Annotated[float, typer.Option(help="The target ratio")] = config.DESIRED_NI_RATIO,
        target_atom_count_importance: Annotated[
            float, typer.Option(help="The importance of getting the atom count right")] = 1,
        target_ratio_importance: Annotated[
            float, typer.Option(help="The importance of getting the ratio right")] = 10000,
        plot: Annotated[bool, typer.Option(help="Whether to plot the nanoparticle")] = False,
        explore_iter: Annotated[int, typer.Option(help="The number of iterations to explore before tuning")] = 5,
        tune_iter: Annotated[int, typer.Option(help="The number of iterations to tune before stopping")] = 25,
):
    """
    Use bayesian search to find the correct value for a parameter in a nanoparticle.\n
    [gray][bold]Note 1[/bold]: The importance of the ratio is much higher because a change in 1 of the ratio is more important than a change of 1 on the atom count.[/gray]
    [gray][bold]Note 2[/bold]: See [yellow]../Shapes/Test/Random_Pores/ironsphere.in[/yellow] for reference.[/gray]
    [gray][bold]Note 3[/bold]: Files in [yellow]../Shapes/Test[/yellow] are ignored by other commands by default because of the different syntax.[/gray]
    """
    rprint(f"Using bayesian search to find the correct value for a parameter in a nanoparticle")
    path = utils.resolve_path(path)
    keys, run_fuzzer = get_full_function(path)
    target_values = [target_atom_count, target_ratio]
    target_importance = [target_atom_count_importance, target_ratio_importance]

    def result(**kwargs):
        result_atom_count, result_ratio, _ = run_fuzzer(**kwargs)
        rmse = compute_rmse(result_atom_count, result_ratio)
        rmse_str = f"{rmse:.4f}"
        kwargs_str = ", ".join([f"'{k}': {v:.4f}" for k, v in kwargs.items()])
        # Todo: Subscribe to Events.OPTIMIZATION_STEP
        rprint(f"{{{kwargs_str}}} ({result_atom_count:5} atoms, {result_ratio:.4f} ratio) RMSE: {rmse_str:14}")
        return -rmse

    def compute_rmse(result_atom_count: list[int], result_ratio: list[float]) -> float:
        return sum(importance * ((actual - target) ** 2) for actual, target, importance in
                   zip((result_atom_count, result_ratio), target_values, target_importance))

    def print_final_result(optim: BayesianOptimization, fuz: Callable):
        atom_count, ratio, nano = fuz(**optim.max['params'])
        resulting_rmse: float = compute_rmse(atom_count, ratio)
        logging.getLogger("").setLevel(config.LOG_LEVEL)
        rprint(optim.max['params'])
        # Todo: Subscribe to Events.OPTIMIZATION_END
        rprint(f"Final result: {atom_count=}, {ratio=} - Weighted RMSE {resulting_rmse}")
        return nano

    param_space = {}
    for key in keys:
        split = key.split(":")
        min_value = float(split[1])
        max_value = float(split[2])
        param_space[key] = (min_value, max_value)
    optimizer = BayesianOptimization(
        f=result,
        pbounds=param_space,
        verbose=0,
        random_state=4,
        allow_duplicate_points=True
    )
    # Disable logging
    logging.getLogger("").setLevel(logging.WARN)
    optimizer.maximize(init_points=explore_iter, n_iter=tune_iter)
    nanoparticle = print_final_result(optimizer, run_fuzzer)
    if plot:
        nanoparticle.plot()
