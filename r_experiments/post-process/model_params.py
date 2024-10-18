import math
from pathlib import Path

import pandas as pd


def generate_path(path: Path):
    all_algo_names = ["ranger", "catboost", "glmnet", "svm"]

    round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))

    def rename_param(param):
        if param.startswith("bestTune."):
            return param[len("bestTune."):]
        return param

    def format_value(value):
        if isinstance(value, float):
            return str(round_to_n(value, 3))
        return str(value)

    def gen_for_algos(algo_names):
        algorithms = " & ".join([
            rf"\multicolumn{{2}}{{|c|}}{{\textbf{{{name}}}}}" for name in algo_names
        ])
        parameters = [
            {key: value for  key, value in dict(pd.read_csv(path / "FeNiMl" / f"{name}_hyperparameters_rmse.csv").iloc[0]).items() if key in ["RMSE_fold", "RMSE_full", "Rsquared_full", "Rsquared_fold"]} for name in algo_names
        ]

        longest_param = max([len(params) for params in parameters])

        values = []
        for item in range(longest_param):
            it = []
            for algo in range(len(algo_names)):
                try:
                    key, value = list(parameters[algo].items())[item]
                    it.extend([fr"\verb|{rename_param(key)}|", str(format_value(value))])
                except IndexError:
                    it.extend(["", ""])
            values.append(" & ".join(it))
        values = " \\\\ \n".join(values)

        return fr"""
        \hline
        {algorithms} \\
        \hline
        {values} \\
        """

    columns = "|".join(["l"] * 2 * 2)
    data = '\n\n'.join([gen_for_algos(names) for names in zip(all_algo_names[::2], all_algo_names[1::2])])
    out = fr"""
    \begin{{table}}[h!]
            \centering
            \caption{{Parameters for each algorithm ({path.name}).}}
            \begin{{tabular}}{{|{columns}|}}
                {data}
                \hline
            \end{{tabular}}
            \label{{tab:algorithm-parameters-{path.name}}}
        \end{{table}}            
    """
    print(out)


def main():
    for path in Path(__file__).parent.parent.iterdir():
        if path.name == "70-30" and (path / "FeNiMl").is_dir():
            generate_path(path)


main()
