This repository contains **extended** code and data for  "Machine learning-based prediction of FeNi nanoparticle magnetization", F. Williamson et al., Journal of Materials Research and Technology (2024). [https://doi.org/10.1016/j.jmrt.2024.10.142](https://doi.org/10.1016/j.jmrt.2024.10.142).

You may find a short version in [this repository](https://github.com/MightyHelper/feni-nanoparticle-magnetization-ml).

# Fe Ni Nanoparticle Simulator

Run a nanoparticle simulation with configurable parametes and execution location.

## Configuration

Check and update [config.py](code/config/config.py) as required.

You may create a `config.local.py` file to override the default values inside `code/`.

You probably need to at least update:
- `LAMMPS_EXECUTABLE`: Your local lammps executable (If you didn't compile lammps manually use `lmp` or the result of `which lmp`)
- `TOKO_USER`: Your username in toko if you want to run simulations there.

Run `pip install -r requirements.txt` inside your desired environment (conda, venv, or other)

Create the executions directory under the project root: `mkdir executions`

## Use

First `cd` into `code`.

- **Linux**: You may use `./cli`
- **Windows & Linux**: Execute `python3 cli.py`

Read the help.

## Explanation

Nanoparticles are located in `Shapes`.

Each nanoparticle consists of a single `.in` file.

This program extracts the lines between the `lattice` command and `# setting` or `mass` lines.
All other lines are discarded.

This can be appreciated with the `nano.shrink` files, which contain the resulting code to be used in the nanoparticle simulation.
Note that the `.in` file is read every time. The `nano.shink` file only exists so that you can see what will be run.
Shrink files can be generated with `./cli sf shrink`.

The program then parses all the commands and regions and normalises them.
The benefit of doing this is that we can modify certain parameters, such as the seed for random operations.
This allows us to generate more nanoparticles with slightly different configuration.

This means that sometimes if using a new command in a nanoparticle, the code might break, as it doesn't know the command.

After generating the region codes for the nanoparticle it is ready to simulate it.

To do so it replaces values into `lammps.template` which allows the program to control parameters such as run length and others.

Afterward, a `SimulationTask` is built.

This `SimulationTask` can be executed on a variety of targets, which may be selected with the `--at` argument where available.

Valid options include
- `local`: Local singlethread execution
- `local:N`: Local multithread execution with N threads
- `toko`: Toko singlethread execution
- `toko:N`: Toko batched execution (We create a single job and execute in batches of N)


Each simulation creates it's own folder under executions.
In there you can find the processed template, and corresponding logs.

The `FeCuNi.eam.alloy` file is always the same for all simulations.

### Running in toko.

To run in toko, please ensure you have access to the `LAMMPS_TOKO_EXECUTABLE` and have created or changed `TOKO_EXECUTION_PATH`.
You may also change the SLURM partition to use with `TOKO_PARTITION_TO_USE`.

The files are:
- Created locally
- Copied to toko
- Dispatched with slurm
- Execution is awaited
- Copied back
- Then analyzed (with ovito etc.)


## Shapes

Shape_Distribution_Interface_Pores_variation.in

Valid Shapes:
- Sphere
- Cone
- Cube
- Ellipsoid
- Cylinder
- Octahedron
- Cross
- Scale
- Toroid

Distribution:
- Janus
  - Axis.Axis       # Axis-aligned separation of Fe and Ni % [X, Y, Z]
  - Corner          # Corner of shape is turned to Ni
- Multilayer.N.Axis # Many Stacked Axis layers % []
- Onion.N[TYPES]           # Stacked layers inwards % [2]
- Multicore.N       # Multiple Ni spheres inside % [3]
- Random            # Random distribution

Sandwich  = Multilayer.3.Axis
CoreShell = Onion.2
MultiShell = Onion.5
(Onion = Onion.7)
PPP       = Corner

Interface
- Mix.N        # Interface is noisy at N % [05, 10]
- Normal       # No noise at interface

Pores
- Full         # Full Shape
- Pores.N[Size]      # N holes inside % [1, 2, 3]

Void = Pores.1
