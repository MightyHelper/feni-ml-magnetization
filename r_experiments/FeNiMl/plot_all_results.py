import glob
import pandas as pd
import matplotlib.pyplot as plt
import typer


def process_csv_files(directory: str, pattern: str) -> dict:
    file_paths = glob.glob(f'{directory}/{pattern}')
    data = {}

    for file_path in file_paths:
        key = file_path.split('/')[-1].split('_')[0]  # Extract key name from file name
        df = pd.read_csv(file_path)
        data[key] = {'RMSE_fold': df['RMSE_fold'].mean(), 'RMSE_full': df['RMSE_full'].mean()}

    return data


def plot_results(data: dict):
    keys = list(data.keys())
    bar_width = 0.35  # Adjust this value according to your preference

    fig, ax = plt.subplots(figsize=(10, 6))
    rmse_fold_values = [data[key]['RMSE_fold'] for key in keys]
    rmse_full_values = [data[key]['RMSE_full'] for key in keys]

    min_rmse_fold_index = rmse_fold_values.index(min(rmse_fold_values))

    ax.barh(keys, rmse_fold_values, bar_width, label='RMSE_fold', color='#5ca61c')  # Soft green
    ax.barh([i + bar_width for i in range(len(keys))], rmse_full_values, bar_width, label='RMSE_full', color='#ffbe30')  # Soft yellow

    ax.set_ylabel('Group')
    ax.set_xlabel('RMSE')
    ax.set_title('Comparison of RMSE_fold and RMSE_full')
    ax.set_yticks([i + bar_width / 2 for i in range(len(keys))])
    ax.set_yticklabels(keys)

    # Annotate the bar with the lowest RMSE_fold value
    ax.scatter(rmse_fold_values[min_rmse_fold_index], min_rmse_fold_index + bar_width / 2,
               color='red', s=100, marker='o', label='Lowest RMSE_fold')
    ax.legend()

    plt.tight_layout()
    plt.show()


def main(directory: str = ".", pattern: str = "*_hyperparameters_rmse.csv"):
    data = process_csv_files(directory, pattern)
    plot_results(data)


if __name__ == "__main__":
    typer.run(main)
