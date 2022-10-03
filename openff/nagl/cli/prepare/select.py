from typing import List


import click
from click_option_group import optgroup

@click.command(
    "select-molecules",
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
    "--n-environment-molecules",
    type=int,
    help="Number of molecules to select from each atom environment",
    default=4,
    show_default=True,
)
@click.option(
    "--element-order",
    type=str,
    multiple=True,
    help="Element order",
    default=["S", "F", "Cl", "Br", "I", "P", "O", "N"],
    show_default=True,
)
def select_molecules_cli(
    input_file: str,
    output_file: str,
    n_environment_molecules: int = 4,
    element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N"]
):
    import tqdm
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.app.partition import DatasetPartitioner

    if not len(input_file) >= 1:
        raise ValueError("At least one input source must be given")

    if isinstance(element_order, str):
        element_order = [x for x in element_order.split() if x]

    all_records = []
    for file in tqdm.tqdm(input_file, desc="loading from stores"):
        store = MoleculeStore(file)
        records = store.retrieve()
        all_records.extend(records)

    partitioner = DatasetPartitioner.from_molecule_records(all_records)

    selected_smiles = partitioner.select_molecules(
        n_environment_molecules=n_environment_molecules,
        element_order=element_order
    )
    selected_records = [
        record
        for record in all_records
        if record.mapped_smiles in selected_smiles
    ]

    destination = MoleculeStore(output_file)
    destination.store(selected_records)

    print(f"Selected {len(selected_records)} molecules from {len(all_records)} original and wrote to {output_file}")



if __name__ == "__main__":
    select_molecules_cli()