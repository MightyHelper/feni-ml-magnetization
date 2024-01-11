import os
import re
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Callable

import nanoparticle_locator
import poorly_coded_parser
import utils
from utils import get_matching

SHAPES = [
    'Sphere',
    'Cone',
    'Cube',
    'Ellipsoid',
    'Cylinder',
    'Octahedron',
]


class NanoparticleRenamer(ABC):
    @classmethod
    def get_all_renames(cls, nanoparticles: list[str]) -> list[tuple[str, str]]:
        nanoparticles = sorted(nanoparticles)
        new_names = []
        out = []
        for old_name in nanoparticles:
            new_name = cls.get_new_nanoparticle_name(old_name, new_names)
            new_names.append(new_name)
            if old_name != new_name:
                out.append((old_name, new_name))
        return out

    @classmethod
    def get_all_renames_for_folder(cls, folder: str = "../Shapes") -> list[tuple[str, str]]:
        nanoparticles = nanoparticle_locator.NanoparticleLocator.sorted_search(folder)
        return cls.get_all_renames(list(nanoparticles))

    @classmethod
    def output_rename(cls, old_name: str, new_name: str) -> None:
        print(f"mv {old_name} {new_name}")

    @classmethod
    def output_renames(cls, renames: list[tuple[str, str]]) -> None:
        for i, (old_name, new_name) in enumerate(renames):
            cls.output_rename(old_name, f"/tmp/deleteme_{i}")
        for i, (old_name, new_name) in enumerate(renames):
            cls.output_rename(f"/tmp/deleteme_{i}", new_name)

    @classmethod
    @abstractmethod
    def get_new_nanoparticle_name(cls, old_name, new_names):
        pass


class BasicNanoparticleRenamer(NanoparticleRenamer):
    @classmethod
    def _standardize_name(cls, old_name: str) -> str:
        norms: list[tuple[str, str]] = [
            (r'Jannus|jannus|janus', 'Janus'),
            (r'Corner|corner', 'Corner'),
            (r'Coreshell|CoreShell', 'CoreShell'),
            (r'Multipores|multipores', 'Multipores'),
        ]
        processed_name: str = old_name
        for norm in norms: processed_name = re.sub(norm[0], norm[1], processed_name)
        return processed_name

    @classmethod
    def _get_sandwich_distribution(cls, processed_name: str) -> str:
        sandwich_types: dict[str, str] = {
            'X': 'Multilayer.3.Axis.X',
            'Y': 'Multilayer.3.Axis.Y',
            'Z': 'Multilayer.3.Axis.Z',
            processed_name: 'Multilayer.3',
        }
        return get_matching(
            sandwich_types,
            processed_name,
            "Couldn't find Janus type in " + processed_name
        )

    @classmethod
    def _find_index(cls, nano_data: dict[str, str | int], other_names: list[str]) -> int:
        index: int = 0
        while index < 1000:
            new_name: str = cls._assemble_nanoparticle_name(index, nano_data)
            if new_name not in other_names:
                return index
            index += 1
        if index == 1000:
            raise Exception("Couldn't find an index for " + str(nano_data) + " in " + str(other_names))

    @classmethod
    def _assemble_nanoparticle_name(cls, index, nano_data):
        parts: list[str] = [
            nano_data['shape'],
            nano_data['distribution'],
            nano_data['interface'],
            nano_data['pores'],
            str(index)
        ]
        return f"{'_'.join(parts)}.in"

    @classmethod
    def _find_interface(cls, processed_name: str) -> str:
        if 'Mix' in processed_name:
            mix_id = re.findall("Mix_?(\d\d)", processed_name)
            if len(mix_id) > 0: return 'Mix.' + str(mix_id[0])
            raise Exception("Couldn't find mix id in " + processed_name)
        else:
            return "Normal"

    @classmethod
    def _get_core_shell_distribution(cls, actual_shape: str, processed_name: str) -> str:
        if processed_name.startswith("../Shapes/CoreShell"):
            rem = processed_name[19:]
            core_shape = min([
                (idx, shape)
                for shape in SHAPES
                if shape != actual_shape and (idx := rem.find(shape)) > -1
            ])[1]
            return f"Onion.2[{core_shape}]"
        return "Onion.2"

    @classmethod
    def _get_janus_distribution(cls, processed_name: str) -> str:
        janus_types: dict[str, str] = {
            'PPP': 'Multilayer.2.Corner',
            'Corner': 'Multilayer.2.Corner',
            'X': 'Multilayer.2.Axis.X',
            'Y': 'Multilayer.2.Axis.Y',
            'Z': 'Multilayer.2.Axis.Z',
        }
        return get_matching(
            janus_types,
            processed_name,
            "Couldn't find Janus type in " + processed_name
        )

    @classmethod
    def _get_distribution(cls, actual_shape: str, processed_name: str) -> str:
        distributions: dict[str, Callable[[], str]] = {
            'Janus': lambda: cls._get_janus_distribution(processed_name),
            'CoreShell': lambda: cls._get_core_shell_distribution(actual_shape, processed_name),
            'Sandwich': lambda: cls._get_sandwich_distribution(processed_name),
            'Multicor': lambda: "Multicore.4",
            'Multishell': lambda: cls._get_onion_levels(processed_name),
            'Onion': lambda: "Onion.7",
            'Multilayer': lambda: "Multilayer.!",
            'Random': lambda: "Random",
        }
        call: Callable[[], str] = get_matching(
            distributions,
            processed_name,
            "Couldn't find distribution in " + processed_name
        )
        return call()

    @classmethod
    def _get_onion_levels(cls, processed_name):
        try:
            path, nano_builder = poorly_coded_parser.PoorlyCodedParser.parse_single_shape(processed_name)
            return f"Onion.{len(nano_builder.regions)}"
        except Exception:
            return "Onion.!"

    @classmethod
    def _get_shape(cls, processed_name: str) -> str:
        try:
            return min([(idx, shape) for shape in SHAPES if (idx := processed_name.find(shape)) > -1])[1]
        except ValueError as e:
            raise Exception("Couldn't find shape in " + processed_name, e)

    @classmethod
    def _get_pores(cls, processed_name: str) -> str:
        if '7.5A' in processed_name:
            return "Pores.1[7.5]"
        elif '8.5A' in processed_name:
            return "Pores.1[8.5]"
        elif 'Multipores' in processed_name:
            n_pores = re.findall("Multipores/(\d)", processed_name)
            if len(n_pores) > 0:
                return f"Pores.{n_pores[0]}"
            else:
                return "Pores.!"
        elif '4sp' in processed_name:
            return "Pores.4[!]"
        elif '3sp' in processed_name:
            return "Pores.3[!]"
        elif '2sp' in processed_name:
            return "Pores.2[!]"
        elif 'Void' in processed_name:
            return "Pores.1[!]"
        else:
            return "Full"

    @classmethod
    def get_new_nanoparticle_name(cls, old_name: str, other_names: list[str]) -> str:
        if os.path.basename(old_name).count("_") > 1 and old_name.endswith(".in"):
            return old_name
        processed_name: str = cls._standardize_name(old_name)
        nano_data: dict[str, str | int] = {
            'shape': cls._get_shape(processed_name),
            'distribution': None,
            'interface': cls._find_interface(processed_name),
            'pores': cls._get_pores(processed_name),
            'index': 0
        }
        nano_data['distribution'] = cls._get_distribution(nano_data['shape'], processed_name)
        nano_data['index'] = cls._find_index(nano_data, other_names)
        return cls._assemble_nanoparticle_name(nano_data['index'], nano_data)


class NewNanoparticleRenamer(NanoparticleRenamer):
    @classmethod
    def get_new_nanoparticle_name(cls, old_name: str, other_names: list[str]) -> str:
        shape, distribution, interface, pores, index = utils.parse_nanoparticle_name(old_name)
        if interface == "05":
            interface = "Mix.05"
        elif interface == "10":
            interface = "Mix.10"
        base_path = os.path.dirname(old_name)
        index = cls.find_index(distribution, interface, old_name, other_names, pores, shape, base_path)
        return cls._assemble_nanoparticle_name(
            shape,
            distribution,
            interface,
            pores,
            index,
            base_path
        )

    @classmethod
    def find_index(cls, distribution, interface, old_name, other_names, pores, shape, base_path):
        index: int = 0
        while index < 1000:
            new_name: str = cls._assemble_nanoparticle_name(
                shape,
                distribution,
                interface,
                pores,
                index,
                base_path
            )
            if new_name not in other_names:
                return index
            index += 1
        if index == 1000:
            raise Exception(f"Couldn\'t find an index for {old_name} in {str(other_names)}")

    @classmethod
    def get_all_renames_for_folder(cls, folder: str = "../Shapes") -> list[tuple[str, str]]:
        nanoparticles = nanoparticle_locator.NanoparticleLocator.sorted_search(folder)
        return cls.get_all_renames(list(nanoparticles))

    @classmethod
    def _assemble_nanoparticle_name(cls, shape, distribution, interface, pores, index, base_path):
        parts: list[str] = [
            shape,
            distribution,
            interface,
            pores,
            str(index)
        ]
        return f"{base_path}/{'_'.join(parts)}.in"


if __name__ == '__main__':
    NewNanoparticleRenamer.output_renames(
        NewNanoparticleRenamer.get_all_renames_for_folder()
    )
