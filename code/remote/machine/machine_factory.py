from pathlib import PurePosixPath

from remote.machine.machine import Machine
from remote.machine.slurm_machine import SLURMMachine


def get_toko_cores(partition: str = "mini") -> int:
    return {
        'debug': 4,
        'mini': 16,
        'Small': 16,
        'prueba': 64,
        'Large': 64,
        'XL': 64,
        'XXL': 128,
    }[partition]


class MachineFactory:
    @staticmethod
    def toko(partition: str, user: str | None = None, node_id: int = 1) -> Machine:
        from config.config import TOKO_URL, TOKO_USER
        user = user if user is not None else TOKO_USER
        return SLURMMachine(
            name=f"toko/{partition}",
            remote_url=TOKO_URL,
            cores=get_toko_cores(partition),
            partition_to_use=partition,
            execution_path=PurePosixPath("~/scratch/projects/magnetism/simulations/"),
            lammps_executable=PurePosixPath("/scratch/fwilliamson/lammps_compile/lammps/build1/lmp"),
            user=user,
            node_id=node_id,
        )
