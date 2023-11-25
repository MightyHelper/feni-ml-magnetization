class MPIOpt:
	use: bool = False
	hw_threads: bool = False
	n_threads: int = 1

	def __init__(self, use: bool = False, hw_threads: bool = False, n_threads: int = 1):
		self.use = use
		self.hw_threads = hw_threads
		self.n_threads = n_threads

	def __str__(self):
		return f"mpirun -n {self.n_threads} {'--use-hwthread-cpus' if self.hw_threads else ''} " if self.use else ""

	def __repr__(self):
		return f"MPIOpt(use={self.use}, hw_threads={self.hw_threads}, n_threads={self.n_threads})"


class GPUOpt:
	use: bool = False

	def __init__(self, use: bool = False):
		self.use = use

	def __str__(self):
		return "-sf gpu -pk gpu 1" if self.use else ""

	def __repr__(self):
		return f"GPUOpt(use={self.use})"


class OMPOpt:
	use: bool = False
	n_threads: int = 1

	def __init__(self, use: bool = False, n_threads: int = 1):
		self.use = use
		self.n_threads = n_threads

	def __str__(self):
		return f"-sf omp -pk omp {self.n_threads}" if self.use else ""

	def __repr__(self):
		return f"OMPOpt(use={self.use}, n_threads={self.n_threads})"
