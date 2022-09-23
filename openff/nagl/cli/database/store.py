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
    from openff.nagl.nn.label import LabelPrecomputedMolecule
    from openff.nagl.storage.record import ConformerRecord, MoleculeRecord
    from openff.nagl.utils.openff import get_coordinates_in_angstrom

    labeller = LabelPrecomputedMolecule(
        partial_charge_method=partial_charge_method,
        bond_order_method=bond_order_method,
    )
    labels = labeller(molecule)

    charges = {}
    if labeller.partial_charge_label in labels:
        charges[partial_charge_method] = labels[labeller.partial_charge_label].numpy()
    bonds = {}
    if labeller.bond_order_label in labels:
        bonds[bond_order_method] = labels[labeller.bond_order_label].numpy()

    conformer_record = ConformerRecord(
        coordinates=get_coordinates_in_angstrom(molecule.conformers[0]),
        partial_charges=charges,
        bond_orders=bonds,
    )
    record = MoleculeRecord(
        mapped_smiles=molecule.to_smiles(mapped=True, isomeric=True),
        conformers=[conformer_record],
    )
    return record


def aggregate_records(
    record_futures: List[Tuple["MoleculeRecord", Optional[str]]]
) -> List[Tuple["MoleculeRecord", Optional[str]]]:
    unsuccessful = []
    seen_smiles = {}

    for record, error in record_futures:
        if error is not None:
            unsuccessful.append((record, error))
            continue

        if record.mapped_smiles in seen_smiles:
            existing = seen_smiles[record.mapped_smiles]
            for conf1 in record.conformers:
                for conf2 in existing.conformers:
                    if (conf1.coordinates - conf2.coordinates).sum() < 0.01:
                        conf2.partial_charges.update(conf1.partial_charges)
                        conf2.bond_orders.update(conf1.bond_orders)
                        break
                else:
                    existing.conformers.append(conf1)

        else:
            seen_smiles[record.mapped_smiles] = record

    results = [(x, None) for x in seen_smiles.values()]
    results.extend(unsuccessful)
    return results


def store_molecules(
    input_file: str,
    output_file: str,
    manager: Optional["Manager"] = None,
    partial_charge_method: str = None,
    bond_order_method: str = None,
):
    from openff.nagl.app.distributed import Manager
    from openff.nagl.cli.utils import (
        as_batch_function_with_captured_errors,
        preprocess_args,
    )
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.utils.openff import stream_molecules_from_file

    manager, input_file, output_file, log_file = preprocess_args(
        manager, input_file, output_file
    )

    molecules = list(stream_molecules_from_file(input_file))
    manager.set_entries(molecules)

    single_func = functools.partial(
        label_precomputed_molecule,
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
            aggregate_function=aggregate_records,
            log_file=log_file,
            n_batches=manager.n_batches,
            desc="storing records",
        )

    print(f"Stored molecules in {output_file}")


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
@click.pass_context
def store_molecules_cli(
    ctx, input_file, output_file, partial_charge_method, bond_order_method
):
    from openff.nagl.cli.utils import get_default_manager

    store_molecules(
        input_file=input_file,
        output_file=output_file,
        manager=get_default_manager(ctx),
        partial_charge_method=partial_charge_method,
        bond_order_method=bond_order_method,
    )


if __name__ == "__main__":
    store_molecules_cli(obj={})
