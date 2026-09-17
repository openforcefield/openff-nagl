import pathlib

import click
import tqdm

from openff.toolkit import Molecule
from openff.units import unit
from openff.nagl import GNNModel


@click.command()
@click.option(
    "--input", "-i",
    "input_directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--output", "-o",
    "output_directory",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--model", "-m",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def main(
    input_directory: str,
    output_directory: str,
    model_path: str,
):
    input_files = sorted(pathlib.Path(input_directory).glob("*.sdf"))
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    model = GNNModel.load(model_path, eval_mode=True)

    for input_file in tqdm.tqdm(input_files):
        mol = Molecule.from_file(input_file, "SDF", allow_undefined_stereo=True)
        mol._partial_charges = (
            model.compute_property(mol, as_numpy=True)
            * unit.elementary_charge
        )
        output_file = output_directory / input_file.name
        mol.to_file(output_file, "SDF")


if __name__ == "__main__":
    main()
