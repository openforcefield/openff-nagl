import functools
from typing import List, Optional, TYPE_CHECKING

import click
import tqdm
from click_option_group import optgroup

if TYPE_CHECKING:
    from openff.nagl.app.distributed import Manager


def get_unique_smiles(file: str, file_format: str = None) -> List[str]:
    from openff.nagl.utils.openff import stream_molecules_from_file

    all_smiles = list(
        tqdm.tqdm(
            stream_molecules_from_file(
                file,
                file_format=file_format,
                as_smiles=True,
            ),
            desc="loading molecules",
            ncols=80,
        )
    )
    unique_smiles = sorted(set(all_smiles))
    n_ignored = len(all_smiles) - len(unique_smiles)

    print(f"{n_ignored} duplicate molecules ignored")
    return unique_smiles


def generate_single_molecule_conformers(
    smiles: str,
    n_conformer_pool: int = 500,
    n_conformers: int = 10,
    rms_cutoff: float = 0.05,
    guess_stereochemistry: bool = True,
):
    from openff.toolkit.topology.molecule import unit as off_unit
    from openff.toolkit.utils.toolkits import RDKitToolkitWrapper
    from openff.nagl.utils.openff import smiles_to_molecule
    import time

    molecule = smiles_to_molecule(smiles, guess_stereochemistry=guess_stereochemistry)

    QC_KWARG = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    molecule.properties["smiles"] = molecule.to_smiles()
    molecule.properties[QC_KWARG] = molecule.to_smiles(mapped=True, isomeric=True)

    molecule.generate_conformers(
        n_conformers=n_conformer_pool,
        rms_cutoff=rms_cutoff * off_unit.angstrom,
        make_carboxylic_acids_cis=True,
    )
    if molecule.conformers is None or not len(molecule.conformers):
        raise ValueError(f"Could not generate conformers for {smiles}")
    try:
        molecule.apply_elf_conformer_selection(limit=n_conformers)
    except RuntimeError as e:
        oe_failure = (
            "OpenEye failed to select conformers, "
            "but did not return any output. "
            "This most commonly occurs when "
            "the Molecule does not have enough conformers to select from"
        )
        if oe_failure in str(e):
            molecule.apply_elf_conformer_selection(
                limit=n_conformers, toolkit_registry=RDKitToolkitWrapper()
            )
        else:
            raise e
    return molecule


def generate_conformers(
    input_file: str,
    output_file: str,
    manager: Optional["Manager"] = None,
    n_conformer_pool: int = 500,
    n_conformers: int = 10,
    conformer_rms_cutoff: float = 0.05,
):
    from openff.nagl.app.distributed import Manager
    from openff.nagl.cli.utils import (
        as_batch_function_with_captured_errors,
        preprocess_args,
    )
    from openff.nagl.utils.openff import stream_molecules_to_file

    manager, input_file, output_file, log_file = preprocess_args(
        manager, input_file, output_file
    )

    unique_smiles = get_unique_smiles(input_file)
    manager.set_entries(unique_smiles)

    single_func = functools.partial(
        generate_single_molecule_conformers,
        n_conformer_pool=n_conformer_pool,
        n_conformers=n_conformers,
        rms_cutoff=conformer_rms_cutoff,
    )
    batch_func = as_batch_function_with_captured_errors(
        single_func, desc="generating conformers"
    )

    with manager:
        futures = manager.submit_to_client(batch_func)

        with stream_molecules_to_file(output_file) as writer:
            manager.store_futures_and_log(
                futures,
                store_function=writer,
                log_file=log_file,
                n_batches=manager.n_batches,
                desc="writing conformers",
            )

    print(f"Wrote molecules in {output_file}")


@click.command("generate-conformers", help="Generate and store conformers")
@click.option(
    "--input-file",
    help="The path to the input molecules. This should either be an SDF or a GZipped "
    "SDF file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-file",
    help="The path to the SDF file (.sdf) to save the generated conformers in.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@optgroup.group("Conformer options")
@optgroup.option(
    "--n-conformer-pool",
    help="The number of conformers to select ELF conformers from",
    type=int,
    default=500,
    show_default=True,
)
@optgroup.option(
    "--n-conformers",
    help="The max number of conformers to select",
    type=int,
    default=10,
    show_default=True,
)
@optgroup.option(
    "--conformer-rms-cutoff",
    help="The RMS cutoff [Å] to use when generating the conformers used for charge "
    "generation.",
    type=float,
    default=0.5,
    show_default=True,
)
@click.pass_context
def generate_conformers_cli(
    ctx, input_file, output_file, n_conformer_pool, n_conformers, conformer_rms_cutoff
):
    from openff.nagl.cli.utils import get_default_manager

    generate_conformers(
        input_file=input_file,
        output_file=output_file,
        manager=get_default_manager(ctx),
        n_conformer_pool=n_conformer_pool,
        n_conformers=n_conformers,
        conformer_rms_cutoff=conformer_rms_cutoff,
    )


if __name__ == "__main__":
    generate_conformers_cli(obj={})
