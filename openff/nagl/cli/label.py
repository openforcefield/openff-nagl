import functools
from typing import TYPE_CHECKING, Optional, Tuple

import click
from click_option_group import optgroup

if TYPE_CHECKING:
    from openff.nagl.app.distributed import Manager
    from openff.nagl.storage.record import MoleculeRecord

def label_molecules(
    input_file: str,
    output_file: str,
    manager: Optional["Manager"] = None,
    partial_charge_methods: Tuple[str] = tuple(),
    bond_order_methods: Tuple[str] = tuple(),
    openeye_only: bool = False,
):

    import tqdm
    from dask import distributed

    from openff.toolkit.utils.toolkits import (
        GLOBAL_TOOLKIT_REGISTRY,
        OpenEyeToolkitWrapper,
    )

    from openff.nagl.app.distributed import Manager
    from openff.nagl.cli.database.store import aggregate_records
    from openff.nagl.cli.utils import (
        as_batch_function_with_captured_errors,
        preprocess_args,
        try_and_return_error,
        write_error_to_file_object
    )
    from openff.nagl.storage.record import MoleculeRecord
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.utils.openff import stream_molecules_from_file

    import tqdm

    manager, input_file, output_file, log_file = preprocess_args(
        manager, input_file, output_file
    )

    molecules = sorted([
        x
        for x in tqdm.tqdm(
            stream_molecules_from_file(input_file, unsafe=True),
            desc="loading molecules",
        )
    ], key=lambda x: x.n_atoms, reverse=True)
    manager.set_entries(molecules)

    if openeye_only:
        registry = OpenEyeToolkitWrapper()
    else:
        registry = GLOBAL_TOOLKIT_REGISTRY

    single_func = functools.partial(
        MoleculeRecord.from_openff,
        partial_charge_methods=partial_charge_methods,
        bond_order_methods=bond_order_methods,
        generate_conformers=False,
        toolkit_registry=registry,
    )
    batch_func = as_batch_function_with_captured_errors(
        single_func, desc="computing labels"
    )

    all_records = []

    with manager as m:
        futures = m.submit_to_client(batch_func)
        with open(log_file, "w") as f:
            for future in tqdm.tqdm(
                distributed.as_completed(futures, raise_errors=False),
                total=m.n_batches,
                desc="gathering records",
                ncols=80,
            ):
                for result, error in future.result():
                    if error is not None:
                        write_error_to_file_object(f, error)
                        continue
                    all_records.append(result)
                
                future.release()

    store = MoleculeStore(output_file)
    store.store(all_records)
    # m.store_futures_and_log(
    #     futures,
    #     store_function=store.store,
    #     aggregate_function=aggregate_records,
    #     log_file=log_file,
    #     n_batches=m.n_batches,
    #     desc="storing records",
    # )

    print(f"Stored molecules in {output_file}")


@click.command("label-molecules", help="Label molecules from SMILES")
@click.option(
    "--input-file",
    help="The path to the input molecules: SDF or smiles. SDFs will be converted to smiles",
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
    help="The partial charge methods to compute",
    multiple=True,
    default=tuple(),
    show_default=True,
)
@optgroup.option(
    "--bond-order-method",
    help="The bond order methods to compute",
    multiple=True,
    default=tuple(),
    show_default=True,
)
@optgroup.option(
    "--openeye-only",
    help="Only use OpenEye",
    is_flag=True,
    type=bool,
    default=False,
    show_default=True,
)
@click.pass_context
def label_molecules_cli(
    ctx,
    input_file,
    output_file,
    partial_charge_method,
    bond_order_method,
    openeye_only
):
    from openff.nagl.cli.utils import get_default_manager

    label_molecules(
        input_file=input_file,
        output_file=output_file,
        manager=get_default_manager(ctx),
        partial_charge_methods=partial_charge_method,
        bond_order_methods=bond_order_method,
        openeye_only=openeye_only,
    )


if __name__ == "__main__":
    label_molecules_cli(obj={})
