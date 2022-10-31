import functools
from typing import TYPE_CHECKING, List, Optional, Tuple

import click
from click_option_group import optgroup

if TYPE_CHECKING:
    from openff.nagl.app.distributed import Manager
    from openff.nagl.storage.record import MoleculeRecord


def label_precomputed_molecule(
    molecule,
    partial_charge_method: str = None,
    bond_order_method: str = None,
):
    import numpy as np
    from openff.toolkit.topology.molecule import unit
    from openff.nagl.storage.record import MoleculeRecord

    if molecule.conformers is None:
        conf = np.zeros((molecule.n_atoms, 3), dtype=float) * unit.angstrom
        molecule.add_conformer(conf)
    return MoleculeRecord.from_precomputed_openff(
        molecule,
        partial_charge_method=partial_charge_method,
        bond_order_method=bond_order_method,
    )


def aggregate_records(
    record_futures: List[Tuple["MoleculeRecord", Optional[str]]]
) -> List[Tuple["MoleculeRecord", Optional[str]]]:
    unsuccessful = []
    seen_smiles = {}
    import numpy as np

    for record, error in record_futures:
        if error is not None:
            unsuccessful.append((record, error))
            continue

        if record.mapped_smiles in seen_smiles:
            existing = seen_smiles[record.mapped_smiles]
            existing.conformers.extend(record.conformers)

        else:
            seen_smiles[record.mapped_smiles] = record

    results = [(seen_smiles.values(), None)] + unsuccessful
    # results = [[(x, None) for x in seen_smiles.values()]]
    # results.extend(unsuccessful)
    return results




def store_molecules(
    input_file: str,
    output_file: str,
    manager: Optional["Manager"] = None,
    partial_charge_method: str = None,
    bond_order_method: str = None,
    allow_empty_molecules: bool = False,
):
    from openff.nagl.storage.record import MoleculeRecord
    from openff.nagl.cli.utils import (
        as_batch_function_with_captured_errors,
        preprocess_args,
    )
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.utils.openff import stream_molecules_from_file

    manager, input_file, output_file, log_file = preprocess_args(
        manager, input_file, output_file
    )

    print(f"Reading from {input_file}")

    molecules = list(stream_molecules_from_file(input_file))
    print(f"Found {len(molecules)} molecules")

    manager.set_entries(molecules)

    if allow_empty_molecules:
        base_func = label_precomputed_molecule
    else:
        base_func = MoleculeRecord.from_precomputed_openff 

    single_func = functools.partial(
        base_func,
        partial_charge_method=partial_charge_method,
        bond_order_method=bond_order_method,
    )
    batch_func = as_batch_function_with_captured_errors(
        single_func, desc="converting to database records"
    )

    with manager:
        futures = manager.submit_to_client(batch_func)

        store = MoleculeStore(output_file)
        manager.store_futures_and_log(
            futures,
            store_function=store.store,
            # aggregate_function=aggregate_records,
            log_file=log_file,
            n_batches=manager.n_batches,
            desc="storing records",
        )

    print(f"Stored molecules in {output_file}")

    retrieved = store.retrieve()
    print(f"{output_file} has {len(retrieved)} records")


    

@click.command("store-molecules", help="Convert pre-computed molecules to database")
@click.option(
    "--input-file",
    help="The path to the input molecules. This should either be an SDF or a GZipped "
    "SDF file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-file",
    help="The path to the SQLite database (.sqlite) to save the labelled molecules in.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@optgroup.group("Labelling options")
@optgroup.option(
    "--partial-charge-method",
    help="The partial charge method used",
    type=str,
    default=None,
    show_default=True,
)
@optgroup.option(
    "--bond-order-method",
    help="The bond order method used",
    type=str,
    default=None,
    show_default=True,
)
@optgroup.option(
    "--allow-empty-molecules",
    help="Whether to allow molecules with no conformers to be stored with zero coordinates",
    is_flag=True,
    default=False,
)
@click.pass_context
def store_molecules_cli(
    ctx, input_file, output_file, partial_charge_method, bond_order_method,
    allow_empty_molecules
):
    from openff.nagl.cli.utils import get_default_manager

    store_molecules(
        input_file=input_file,
        output_file=output_file,
        manager=get_default_manager(ctx),
        partial_charge_method=partial_charge_method,
        bond_order_method=bond_order_method,
        allow_empty_molecules=allow_empty_molecules,
    )


if __name__ == "__main__":
    store_molecules_cli(obj={})
