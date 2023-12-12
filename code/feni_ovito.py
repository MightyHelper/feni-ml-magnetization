import logging
import numpy
from multiprocessing import Process

XYZ_FILENAME = "XYZ.xyz"
G_R_FILENAME = 'g(r).txt'
PARTIAL_G_R_FILENAME = 'Partial_g(r).txt'
SURFACE_FILENAME = "Surface_atoms.txt"
PROPORTION_FILENAME = "Type_proportion.txt"
COORD_NI_FILENAME = "Coordination_Histogram_Ni.txt"
COORD_FE_FILENAME = "Coordination_Histogram_Fe.txt"
PEH_FILENAME = "PotentialEnergy_Histogram.txt"
COORD_FILENAME = "Coordination_Histogram.txt"


class FeNiOvitoParser:
	@staticmethod
	def __parse_worker(
		filenames=None
	):
		from ovito.io import import_file, export_file
		from ovito.modifiers import SelectTypeModifier, DeleteSelectedModifier, CoordinationAnalysisModifier, HistogramModifier, ExpressionSelectionModifier
		filenames = {
			'dump': 'iron.300000.dump',
			'xyz': XYZ_FILENAME,
			'gr': G_R_FILENAME,
			'grp': PARTIAL_G_R_FILENAME,
			'coordh': COORD_FILENAME,
			'peh': PEH_FILENAME,
			'coordhfe': COORD_FE_FILENAME,
			'coordhni': COORD_NI_FILENAME,
			'proportion': PROPORTION_FILENAME,
			'surface': SURFACE_FILENAME,
			'base_path': "",
			**filenames
		}
		base_path = filenames['base_path']
		dump = base_path + filenames['dump']
		xyz = base_path + filenames['xyz']
		gr = base_path + filenames['gr']
		grp = base_path + filenames['grp']
		coordh = base_path + filenames['coordh']
		peh = base_path + filenames['peh']
		coordhfe = base_path + filenames['coordhfe']
		coordhni = base_path + filenames['coordhni']
		proportion = base_path + filenames['proportion']
		surface = base_path + filenames['surface']

		data = import_file(dump)
		datatemp = data.compute()
		natoms = datatemp.particles.count
		export_file(data, xyz, "xyz", columns=['Particle Type', 'Position.X', 'Position.Y', 'Position.Z'])
		data.modifiers.append(CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=100, partial=False))
		datasave = data.compute()
		numpy.savetxt(gr, datasave.tables['coordination-rdf'].xy(), header="Radial distribution function:\n \"Pair separation distance\" g(r)")

		data = import_file(dump)
		data.modifiers.append(CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=100, partial=True))
		datasave = data.compute()
		numpy.savetxt(grp, datasave.tables['coordination-rdf'].xy(), header="Radial distribution function:\n \"Pair separation distance\" 1-1 1-2 2-2")

		data = import_file(dump)
		coordination_analysis = CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100)
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(HistogramModifier(bin_count=10, property='Coordination', fix_xrange=True, xrange_start=0.0, xrange_end=10))
		export_file(data, coordh, "txt/series", key="histogram[Coordination]")
		data.modifiers.append(HistogramModifier(bin_count=100, property='c_peatom', fix_xrange=True, xrange_start=-5.0, xrange_end=-2.0))
		export_file(data, peh, "txt/series", key="histogram[c_peatom]")

		data = import_file(dump)
		data.modifiers.append(SelectTypeModifier(types={1}))
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(HistogramModifier(bin_count=10, property='Coordination', fix_xrange=True, xrange_start=0.0, xrange_end=10, only_selected=True))
		export_file(data, coordhfe, "txt/series", key="histogram[Coordination]")

		data = import_file(dump)
		data.modifiers.append(SelectTypeModifier(types={2}))
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(HistogramModifier(bin_count=10, property='Coordination', fix_xrange=True, xrange_start=0.0, xrange_end=10, only_selected=True))
		export_file(data, coordhni, "txt/series", key="histogram[Coordination]")

		data = import_file(dump)
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(ExpressionSelectionModifier(expression='Coordination>7'))
		data.modifiers.append(DeleteSelectedModifier())
		data.modifiers.append(SelectTypeModifier(types={1}))
		dataFe = data.compute()
		nFeShell = dataFe.attributes['SelectType.num_selected']

		data = import_file(dump)
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(ExpressionSelectionModifier(expression='Coordination>7'))
		data.modifiers.append(DeleteSelectedModifier())
		data.modifiers.append(SelectTypeModifier(types={2}))
		dataNi = data.compute()
		nNiShell = dataNi.attributes['SelectType.num_selected']

		data = import_file(dump)
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(ExpressionSelectionModifier(expression='Coordination<=7'))
		data.modifiers.append(DeleteSelectedModifier())
		data.modifiers.append(SelectTypeModifier(types={1}))
		dataFe = data.compute()
		nFeCore = dataFe.attributes['SelectType.num_selected']

		data = import_file(dump)
		data.modifiers.append(coordination_analysis)
		data.modifiers.append(ExpressionSelectionModifier(expression='Coordination<=7'))
		data.modifiers.append(DeleteSelectedModifier())
		data.modifiers.append(SelectTypeModifier(types={2}))
		dataNi = data.compute()
		nNiCore = dataNi.attributes['SelectType.num_selected']
		data = numpy.char.add(str(natoms), "   ")
		data = numpy.char.add(data, str(round(nFeShell / natoms, 3)))
		data = numpy.char.add(data, "   ")
		data = numpy.char.add(data, str(round(nNiShell / natoms, 3)))
		data = numpy.char.add(data, "   ")
		data = numpy.char.add(data, str(round(nFeCore / natoms, 3)))
		data = numpy.char.add(data, "   ")
		data = numpy.char.add(data, str(round(nNiCore / natoms, 3)))
		data = numpy.array([data, " "])
		numpy.savetxt(surface, data, fmt='%s', header="Ntotal N_Fe_surf / Ntotal N_Ni_surf / Ntotal N_Fe_core / Ntotal N_Ni_core / Ntotal")

		data = import_file(dump)
		data.modifiers.append(SelectTypeModifier(types={1}))
		dataFe = data.compute()
		nFe = dataFe.attributes['SelectType.num_selected']

		data = import_file(dump)
		data.modifiers.append(SelectTypeModifier(types={2}))
		dataNi = data.compute()
		nNi = dataNi.attributes['SelectType.num_selected']

		data = numpy.char.add(str(round(nFe / natoms, 3)), "   ")
		data = numpy.char.add(data, str(round(nNi / natoms, 3)))
		data = numpy.array([data, " "])
		numpy.savetxt(proportion, data, fmt='%s', header="N_Fe / Ntotal N_Ni / Ntotal")

	@staticmethod
	def parse(filenames=None):
		filenames = {} if filenames is None else filenames
		p = Process(target=FeNiOvitoParser.__parse_worker, args=(filenames,))
		p.start()
		logging.info("Waiting for OVITO...")
		p.join()  # this blocks until the process terminates
		logging.info("Done")


if __name__ == '__main__':
	FeNiOvitoParser.parse()
