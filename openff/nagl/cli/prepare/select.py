import json
import pathlib
from typing import List

import click
from click_option_group import optgroup


@click.command(
    "select",
    short_help="Select broad set of chemistries from dataset.",
    help=(
        "Selects a set of molecules based on the criteria specified by:\n\n"
        "    [1] Bleiziffer, Patrick, Kay Schaller, and Sereina Riniker. 'Machine learning "
        "of partial charges derived from high-quality quantum-mechanical calculations.' "
        "JCIM 58.3 (2018): 579-590.\n\n"
    ),
)
@click.option(
    "--input-file",
    help="The path to the input molecules (.sqlite)",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option(
    "--output-file",
    help="The path (.sqlite) to save the filtered molecules to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--n-min-molecules",
    type=int,
    help="Minimum number of molecules to select from each atom environment",
    default=4,
    show_default=True,
)
@click.option(
    "--element-order",
    type=str,
    multiple=True,
    help="Element order",
    default=["S", "F", "Cl", "Br", "I", "P", "O", "N", "C"],
    show_default=True,
)
@click.option(
    "--output-source-file",
    help="The path (.json) to save the source information for data.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="selected-data-sources.json",
    show_default=True,
)
@click.option(
    "--clean-filenames",
    help="If on, only save the base filename instead of the whole path",
    is_flag=True,
    default=True,
    show_default=True,
)
def select_molecules_cli(
    input_file: str,
    output_file: str,
    n_min_molecules: int = 4,
    element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N", "C"],
    output_source_file: str = "selected-data-sources.json",
    clean_filenames: bool = True,
):
    import tqdm

    from openff.nagl.app.partitioner import DatasetPartitioner
    from openff.nagl.storage.store import MoleculeStore

    if not len(input_file) >= 1:
        raise ValueError("At least one input source must be given")

    if isinstance(element_order, str):
        element_order = [x for x in element_order.split() if x]

    smiles_to_records = {}
    smiles_to_db = {}
    for file in tqdm.tqdm(input_file, desc="loading from stores"):
        store = MoleculeStore(file)
        records = store.retrieve()
        if clean_filenames:
            file = pathlib.Path(file).stem
        for rec in records:
            smiles_to_records[rec.mapped_smiles] = rec
            smiles_to_db[rec.mapped_smiles] = file

    partitioner = DatasetPartitioner(smiles_to_db)

    selected_data = partitioner.select_atom_environments(
        n_min_molecules=n_min_molecules, element_order=element_order
    )
    selected_records = [smiles_to_records[x] for x in selected_data.labelled_smiles]

    destination = MoleculeStore(output_file)
    destination.store(selected_records)

    print(
        f"Selected {len(selected_records)} molecules from {len(smiles_to_records)} original and wrote to {output_file}"
    )

    with open(output_source_file, "w") as f:
        json.dump(selected_data.labelled_smiles, f)


if __name__ == "__main__":
    select_molecules_cli()
