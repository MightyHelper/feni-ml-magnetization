import numpy
from ovito.io import import_file, export_file
from ovito.modifiers import SelectTypeModifier, DeleteSelectedModifier, CoordinationAnalysisModifier, HistogramModifier, ExpressionSelectionModifier

data = import_file('iron.300000.dump')
datatemp = data.compute()
natoms = datatemp.particles.count
export_file(data, "XYZ.xyz", "xyz", columns=['Particle Type', 'Position.X', 'Position.Y', 'Position.Z'])
data.modifiers.append(CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=100, partial=False))
datasave = data.compute()
numpy.savetxt('g(r).txt', datasave.tables['coordination-rdf'].xy(), header="Radial distribution function:\n \"Pair separation distance\" g(r)")

data = import_file('iron.300000.dump')
data.modifiers.append(CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=100, partial=True))
datasave = data.compute()
numpy.savetxt('Partial_g(r).txt', datasave.tables['coordination-rdf'].xy(), header="Radial distribution function:\n \"Pair separation distance\" 1-1 1-2 2-2")

data = import_file('iron.300000.dump')
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
data.modifiers.append(HistogramModifier(bin_count=100, property='Coordination', fix_xrange=True, xrange_start=0.0, xrange_end=10))
export_file(data, "Coordination_Histogram.txt", "txt/series", key="histogram[Coordination]")
data.modifiers.append(HistogramModifier(bin_count=100, property='c_peatom', fix_xrange=True, xrange_start=-5.0, xrange_end=-2.0))
export_file(data, "PotentialEnergy_Histogram.txt", "txt/series", key="histogram[c_peatom]")

data = import_file('iron.300000.dump')
data.modifiers.append(SelectTypeModifier(types={1}))
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
data.modifiers.append(HistogramModifier(bin_count=100, property='Coordination', fix_xrange=True, xrange_start=0.0, xrange_end=10, only_selected=True))
export_file(data, "Coordination_Histogram_Fe.txt", "txt/series", key="histogram[Coordination]")

data = import_file('iron.300000.dump')
data.modifiers.append(SelectTypeModifier(types={2}))
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
data.modifiers.append(HistogramModifier(bin_count=100, property='Coordination', fix_xrange=True, xrange_start=0.0, xrange_end=10, only_selected=True))
export_file(data, "Coordination_Histogram_Ni.txt", "txt/series", key="histogram[Coordination]")

data = import_file('iron.300000.dump')
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
data.modifiers.append(ExpressionSelectionModifier(expression='Coordination>7'))
data.modifiers.append(DeleteSelectedModifier())
data.modifiers.append(SelectTypeModifier(types={1}))
dataFe = data.compute()
nFeShell = dataFe.attributes['SelectType.num_selected']

data = import_file('iron.300000.dump')
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
data.modifiers.append(ExpressionSelectionModifier(expression='Coordination>7'))
data.modifiers.append(DeleteSelectedModifier())
data.modifiers.append(SelectTypeModifier(types={2}))
dataNi = data.compute()
nNiShell = dataNi.attributes['SelectType.num_selected']

data = import_file('iron.300000.dump')
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
data.modifiers.append(ExpressionSelectionModifier(expression='Coordination<=7'))
data.modifiers.append(DeleteSelectedModifier())
data.modifiers.append(SelectTypeModifier(types={1}))
dataFe = data.compute()
nFeCore = dataFe.attributes['SelectType.num_selected']

data = import_file('iron.300000.dump')
data.modifiers.append(CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=100))
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
numpy.savetxt("Surface_atoms.txt", data, fmt='%s', header="Ntotal N_Fe_surf / Ntotal N_Ni_surf / Ntotal N_Fe_core / Ntotal N_Ni_core / Ntotal")

data = import_file('iron.300000.dump')
data.modifiers.append(SelectTypeModifier(types={1}))
dataFe = data.compute()
nFe = dataFe.attributes['SelectType.num_selected']

data = import_file('iron.300000.dump')
data.modifiers.append(SelectTypeModifier(types={2}))
dataNi = data.compute()
nNi = dataNi.attributes['SelectType.num_selected']

data = numpy.char.add(str(round(nFe / natoms, 3)), "   ")
data = numpy.char.add(data, str(round(nNi / natoms, 3)))
data = numpy.array([data, " "])
numpy.savetxt("Type_proportion.txt", data, fmt='%s', header="N_Fe / Ntotal N_Ni / Ntotal")
