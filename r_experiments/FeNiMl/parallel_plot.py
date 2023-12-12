# from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parallel_coordinates(
	frame, class_column, cols=None, ax=None, color=None,
	use_columns=False, xticks=None, colormap=None,
	**kwds):
	import matplotlib.pyplot as plt
	import matplotlib as mpl

	n = len(frame)
	class_col = frame[class_column]
	class_min = np.amin(class_col)
	class_max = np.amax(class_col)

	if cols is None:
		df = frame.drop(class_column, axis=1)
	else:
		df = frame[cols]

	ncols = len(df.columns)

	# determine values to use for xticks
	if use_columns is True:
		if not np.all(np.isreal(list(df.columns))):
			raise ValueError('Columns must be numeric to be used as xticks')
		x = df.columns
	elif xticks is not None:
		if not np.all(np.isreal(xticks)):
			raise ValueError('xticks specified must be numeric')
		elif len(xticks) != ncols:
			raise ValueError('Length of xticks must match number of columns')
		x = xticks
	else:
		x = range(ncols)

	fig = plt.figure()
	ax = plt.gca()

	Colorm = plt.get_cmap(colormap)

	for i in range(n):
		y = df.iloc[i].values
		kls = class_col.iat[i]
		ax.plot(x, y, color=Colorm((kls - class_min) / (class_max - class_min)), **kwds)

	for i in x:
		ax.axvline(i, linewidth=1, color='black')

	ax.set_xticks(x)
	ax.set_xticklabels(df.columns)
	ax.set_xlim(x[0], x[-1])
	ax.legend(loc='upper right')
	ax.grid()

	bounds = np.linspace(class_min, class_max, 10)
	cax, _ = mpl.colorbar.make_axes(ax)
	cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

	return fig


df = pd.read_csv("catboost.csv")
print(df.columns)
df.drop(columns=["Unnamed: 0", "Rsquared", "MAE", "MAESD", "RMSESD", "RsquaredSD"], inplace=True)


# Normalize each column from 0 to 1
# df = (df - df.min()) / (df.max() - df.min())
def apply_col(df, col):
	# if is number
	if np.issubdtype(df[col].dtype, np.number):
		if (np.max(df[col]) - np.min(df[col])) == 0:
			df[col] = 0.5
			return
		df[col] = (df[col] - np.min(df[col])) / (np.max(df[col]) - np.min(df[col]))
	else:
		# Encode categorical columns
		all_values = df[col].unique()
		# Convert to index
		df[col] = df[col].apply(lambda x: np.where(all_values == x)[0][0])


column_names = df.columns.values.tolist()
# Columns with single value
columns_with_single_value = [col for col in column_names if len(df[col].unique()) == 1]
print("Columns with single value: ", columns_with_single_value)
df = df.drop(columns=columns_with_single_value)
column_names = df.columns.values.tolist()
for col in column_names:
	apply_col(df, col)
df["RMSE_"] = df["RMSE"]
print(df)
parallel_coordinates(df, 'RMSE')
plt.show()
