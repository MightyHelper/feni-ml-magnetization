import numpy as np


def extract_magnetism(in_log='log.lammps', out_log='log.out.lammps', out_mag='Magnetization.txt'):
	full_line = " "
	new_file = np.array(['#datos'])
	with open(in_log, "r") as in_file:
		while full_line != "":
			full_line = in_file.readline()
			tmp = full_line.replace(' ', '').replace('.', '').replace('-', '')
			try:
				float(tmp.strip())
				new_file = np.vstack((new_file, full_line.strip()))
			except:
				pass
	np.savetxt(out_log, new_file, fmt='%s')

	# Separate data in columns
	heading = ["Step", "Temp", "v_tmag", "v_magx", "v_magy", "v_magz", "v_magnorm", "KinEng", "PotEng", "TotEng", "Press"]
	datos = np.genfromtxt(out_log, names=heading, dtype='f', skip_header=1)

	# Calculate mean and std magnetization of last 100.000 steps
	mag_total = np.around(np.mean(datos['v_magnorm'][np.size(datos['v_magnorm']) - 201:-1]), 2).astype(str)
	mag_error = np.around(np.std(datos['v_magnorm'][np.size(datos['v_magnorm']) - 201:-1]), 2).astype(str)
	mag = mag_total + ' ' + mag_error
	mag = np.vstack((mag, ' '))
	np.savetxt(out_mag, mag, fmt='%s', header='#TotMag Error')


if __name__ == '__main__':
	extract_magnetism("../executions/log.lammps")
