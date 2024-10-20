from pathlib import PurePosixPath

from remote.machine.machine import Machine
from remote.machine.slurm_machine import SLURMMachine


def get_toko_cores(partition: str = "mini") -> int:
    return {
        'debug': 4,
        'mini': 16,
        'Small': 16,
        'prueba': 64,
        'Large': 32,
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
            execution_path=PurePosixPath(f"/scratch/{TOKO_USER}/projects/magnetism/simulations/"),
            lammps_executable=PurePosixPath("/scratch/fwilliamson/lammps_compile/lammps/build2/lmp"),
            user=user,
            node_id=node_id,
            launch_time=0.0,  # TODO: revise
            single_core_completion_time=(4 * 3155.91) / (3.14 if partition == "mini" else 1)  # TODO: revise
        )
