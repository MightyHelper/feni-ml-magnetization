import re

import rich

import nanoparticle_locator
import poorly_coded_parser
import utils

nanoparticles = nanoparticle_locator.NanoparticleLocator.sorted_search("../Shapes")
renames = {

}
shapes = [
    'Sphere',
    'Cone',
    'Cube',
    'Ellipsoid',
    'Cylinder',
    'Octahedron',
]
for nanoparticle in nanoparticles:
    oldname = nanoparticle
    nanoparticle = re.sub(r'Jannus|jannus|janus', 'Janus', nanoparticle)
    nanoparticle = re.sub(r'Corner|corner', 'Corner', nanoparticle)
    nanoparticle = re.sub(r'Coreshell|CoreShell', 'CoreShell', nanoparticle)
    nanoparticle = re.sub(r'Multipores|multipores', 'Multipores', nanoparticle)
    nano_data = {
        'shape': None,
        'distribution': None,
        'interface': None,
        'pores': None,
        'index': 0
    }
    if '7.5A' in nanoparticle:
        nano_data['pores'] = "Pores.1[7.5]"
    elif '8.5A' in nanoparticle:
        nano_data['pores'] = "Pores.1[8.5]"
    elif 'Multipores' in nanoparticle:
        n_pores = re.findall("Multipores/(\d)", nanoparticle)
        if len(n_pores) > 0:
            nano_data['pores'] = f"Pores.{n_pores[0]}"
        else:
            nano_data['pores'] = "Pores.?"
    else:
        nano_data['pores'] = "Full"
    # First shape that occurs
    try:
        nano_data['shape'] = min([(idx, shape) for shape in shapes if (idx:=nanoparticle.find(shape)) > -1])[1]
    except ValueError as e:
        print(nanoparticle)
        print(e)
        exit(-1)

    if 'Janus' in nanoparticle:
        if 'PPP' in nanoparticle or 'Corner' in nanoparticle:
            nano_data['distribution'] = "Multilayer.2.Corner"
        elif "X" in nanoparticle:
            nano_data['distribution'] = "Multilayer.2.Axis.X"
        elif "Y" in nanoparticle:
            nano_data['distribution'] = "Multilayer.2.Axis.Y"
        elif "Z" in nanoparticle:
            nano_data['distribution'] = "Multilayer.2.Axis.Z"
        else:
            print(">>", nanoparticle)

    if 'CoreShell' in nanoparticle:
        nano_data['distribution'] = "Onion.2"
        if nanoparticle.startswith("../Shapes/CoreShell"):
            rem = nanoparticle[19:]
            core_shape = min([(idx, shape) for shape in shapes if shape != nano_data['shape'] and (idx:=nanoparticle.find(shape)) > -1])[1]
            nano_data['distribution'] = f"Onion.2[{core_shape}]"
    elif 'Sandwich' in nanoparticle:
        nano_data['distribution'] = "Sandwich"
    elif 'Multicor' in nanoparticle:
        nano_data['distribution'] = "Multicore.4"
    elif 'Multishell' in nanoparticle:
        nano_data['distribution'] = "Onion.?"
        try:
            path, nano_builder = poorly_coded_parser.PoorlyCodedParser.parse_single_shape(nanoparticle)
            nano_data['distribution'] = f"Onion.{len(nano_builder.regions)}"
        except Exception:
            pass
    elif 'Onion' in nanoparticle:
        nano_data['distribution'] = "Onion.7"
    elif 'Multilayer' in nanoparticle:
        nano_data['distribution'] = "Multilayer"
    elif nano_data['distribution'] is None and 'Random' in nanoparticle:
        nano_data['distribution'] = "Random"

    if 'Mix' in nanoparticle:
        mix_id = re.findall("Mix_?(\d\d)", nanoparticle)
        if len(mix_id) > 0:
            nano_data['interface'] = mix_id[0]
        else:
            print("ERR")
            exit(-1)
    else:
        nano_data['interface'] = "Normal"

    index = 0
    while True:
        newname = f"{nano_data['shape']}_{nano_data['distribution']}_{nano_data['interface']}_{nano_data['pores']}_{index}.in"
        if newname not in renames:
            renames[newname] = (oldname, nanoparticle)
            break
        index += 1


rich.print(renames)