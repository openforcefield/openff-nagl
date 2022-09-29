from typing import Tuple

import click
from click_option_group import optgroup


@click.command(
    "filter",
    short_help="Filter undesirable chemistries and counter-ions.",
    help="Filters a set of molecules based on the criteria specified by:\n\n"
    "    [1] Bleiziffer, Patrick, Kay Schaller, and Sereina Riniker. 'Machine learning "
    "of partial charges derived from high-quality quantum-mechanical calculations.' "
    "JCIM 58.3 (2018): 579-590.\n\nIn particular molecules are only retained if they "
    "have a weight between 250 and 350 g/mol, have less than seven rotatable bonds and "
    "are composed of only H, C, N, O, F, P, S, Cl, Br, and I.\n\nThis script will also "
    "optionally remove any counter-ions by retaining only the largest molecule if "
    "multiple components are present.",
)
@click.option(
    "--input-file",
    help="The path to the input molecules: SDF, zipped SDF, or smiles file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-file",
    help="The path to save the filtered molecules to. This should be an SDF or smiles file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--only-retain-largest",
    is_flag=True,
    help="If specified counter ions (and molecules) will be removed.",
    default=False,
    show_default=True,
)
@click.option(
    "--min-mass",
    type=float,
    help="Minimum mass (g/mol)",
    default=250,
    show_default=True,
)
@click.option(
    "--max-mass",
    type=float,
    help="Maximum mass (g/mol)",
    default=250,
    show_default=True,
)
@click.option(
    "--n-rotatable-bonds",
    type=int,
    help="Number of rotatable bonds",
    default=7,
    show_default=True,
)
@click.option(
    "--element",
    type=str,
    multiple=True,
    help="Allowed elements",
    default=("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"),
    show_default=True,
)
@optgroup.group("Parallelization configuration")
@optgroup.option(
    "--n-processes",
    help="The number of processes to parallelize the filtering over.",
    type=int,
    default=1,
    show_default=True,
)
def filter_cli(
    input_file: str,
    output_file: str,
    element: Tuple[str, ...],
    only_retain_largest: bool,
    min_mass: float = 250,
    max_mass: float = 350,
    n_rotatable_bonds: int = 7,
    n_processes: int = 1,
):
    import tqdm

    from openff.nagl.app.filter import filter_molecules
    from openff.nagl.utils.openff import (
        stream_molecules_to_file,
        stream_molecules_from_file
    )

    with stream_molecules_to_file(output_file) as writer:
        for molecule in tqdm.tqdm(
            filter_molecules(
                stream_molecules_from_file(input_file),
                only_retain_largest=only_retain_largest,
                allowed_elements=element,
                min_mass=min_mass,
                max_mass=max_mass,
                n_rotatable_bonds=n_rotatable_bonds,
                n_processes=n_processes,
            ),
            desc="Writing molecules",
        ):
            writer(molecule)
            

if __name__ == "__main__":
    filter_cli()
