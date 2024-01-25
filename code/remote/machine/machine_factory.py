from pathlib import PurePosixPath

from remote.machine.machine import Machine
from remote.machine.slurm_machine import SLURMMachine


def get_toko_cores(partition: str = "mini") -> int:
    return {
        'mini': 16,
        'XL': 64,
        'XXL': 128,
    }[partition]


class MachineFactory:
    @staticmethod
    def toko(partition: str, user: str | None = None, node_id: int = 1, copy_script: str = 'rsync') -> Machine:
        from config.config import TOKO_URL, TOKO_USER
        user = user if user is not None else TOKO_USER
        return SLURMMachine(
            name="toko",
            remote_url=TOKO_URL,
            cores=get_toko_cores(partition),
            partition_to_use=partition,
            copy_script=PurePosixPath(copy_script),
            execution_path=PurePosixPath("/scratch/fwilliamson/projects/magnetism/simulations/"),
            lammps_executable=PurePosixPath("/scratch/fwilliamson/lammps_compile/lammps/build1/lmp"),
            user=user,
            node_id=node_id,
        )
