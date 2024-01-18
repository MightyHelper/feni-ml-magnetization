from remote.machine import Machine
from remote.slurm_machine import SLURMMachine


def get_toko_cores(partition: str = "mini") -> int:
    return {
        'mini': 16,
        'XL': 64,
        'XXL': 128,
    }[partition]


class MachineFactory:
    @staticmethod
    def toko(partition: str, node_id: int = 1) -> Machine:
        from config import TOKO_URL, TOKO_USER
        return SLURMMachine(
            name="toko",
            remote_url=TOKO_URL,
            cores=get_toko_cores(partition),
            partition_to_use=partition,
            user=TOKO_USER,
            node_id=node_id,
        )
