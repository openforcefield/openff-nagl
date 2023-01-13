from typing import TYPE_CHECKING, Optional

import click
from click_option_group import optgroup

if TYPE_CHECKING:
    from openff.nagl.app.distributed import Manager


def retrieve_molecules(
    input_file: str,
    output_file: str,
    manager: Optional["Manager"] = None,
    partial_charge_method: Optional[str] = None,
    bond_order_method: Optional[str] = None,
):
    from openff.nagl.cli.utils import (
        as_batch_function_with_captured_errors,
        preprocess_args,
    )
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.utils.openff import stream_molecules_to_file

    manager, input_file, output_file, log_file = preprocess_args(
        manager, input_file, output_file
    )

    store = MoleculeStore(str(input_file))
    records = store.retrieve(
        partial_charge_methods=partial_charge_method,
        bond_order_methods=bond_order_method,
    )
    manager.set_entries(records)

    def single_func(record):
        record = record.to_openff(
            partial_charge_method=partial_charge_method,
            bond_order_method=bond_order_method,
            normalize_partial_charges=False
        )
        return record

    batch_func = as_batch_function_with_captured_errors(
        single_func,
        desc="retrieving molecules",
    )

    with manager:
        futures = manager.submit_to_client(batch_func)
        with stream_molecules_to_file(output_file) as writer:
            manager.store_futures_and_log(
                futures,
                store_function=writer,
                log_file=log_file,
                n_batches=manager.n_batches,
                desc="saving molecules",
            )
        print(f"Wrote molecules to {output_file}")


@click.command("retrieve-molecules", help="Retrieve molecules from database")
@click.option(
    "--input-file",
    help="The path to the SQLite database (.sqlite) to retrieve the labelled molecules from.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-file",
    help="The path to the file to save the molecules in. This should be an SDF file.",
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
def retrieve_molecules_cli(
    ctx, input_file, output_file, partial_charge_method, bond_order_method
):
    from openff.nagl.cli.utils import get_default_manager

    retrieve_molecules(
        input_file=input_file,
        output_file=output_file,
        manager=get_default_manager(ctx),
        partial_charge_method=partial_charge_method,
        bond_order_method=bond_order_method,
    )


if __name__ == "__main__":
    retrieve_molecules_cli(obj={})
