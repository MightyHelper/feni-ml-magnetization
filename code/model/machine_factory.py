import multiprocessing

from model.machine import Machine, SLURMMachine


def get_toko_cores(partition: str = "mini") -> int:
    return {
        'mini': 16,
        'XL': 64,
        'XXL': 128,
    }[partition]


class MachineFactory:
    @staticmethod
    def local() -> Machine:
        return Machine(
            name="local",
            cores=multiprocessing.cpu_count(),
        )

    @staticmethod
    def toko(partition: str, node_id: int = 1) -> Machine:
        from config import TOKO_URL, TOKO_USER
        return SLURMMachine(
            name="toko",
            hostname=TOKO_URL,
            cores=get_toko_cores(partition),
            user=TOKO_USER,
            partition=partition,
            node_id=node_id,
        )
