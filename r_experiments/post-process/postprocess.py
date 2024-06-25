import pandas as pd
import os
from pathlib import Path

def calc_complete(root_path, name):
    files: list[tuple[str, pd.DataFrame]] = [(file.rsplit("_")[0], pd.read_csv(root_path / file)) for file in os.listdir(root_path) if file.endswith("rmse.csv")]
    complete_df = pd.DataFrame([{'method': method, 'fold': df.iloc[0]['RMSE_fold'], 'full': df.iloc[0]['RMSE_full']} for (method, df) in files])
    complete_df = complete_df.sort_values('method').reset_index(drop=True)
    complete_df['experiment'] = name
    print(complete_df.sort_values('fold'), end="\n\n")
    complete_df.to_csv(root_path / "full_results.csv", index=False)
    return complete_df

sub_paths = ["60-40", "70-30", "no-split"]
dfs = []
for sub_path in sub_paths:
    dfs.append(calc_complete(Path(sub_path) / "FeNiMl", sub_path).reset_index(drop=True))

all_experiments = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
print(all_experiments.sort_values('fold'))

all_experiments.to_csv("all_experiment_results.csv", index=False)
