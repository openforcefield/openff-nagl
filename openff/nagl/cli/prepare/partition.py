from typing import List


import click

@click.command(
    "partition-molecules",
    help="Partition molecules into training, validation, test datasets",
)
@click.option(
    "--input-file",
    help="The path to the input molecules (.sqlite)",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option(
    "--training-fraction",
    help="Rough percentage of how much should be in the training set.",
    type=float,
    default=0.7,
    show_default=True,
)
@click.option(
    "--validation-fraction",
    help="Rough percentage of how much should be in the validation set.",
    type=float,
    default=0.2,
    show_default=True,
)
@click.option(
    "--test-fraction",
    help="Rough percentage of how much should be in the test set.",
    type=float,
    default=0.1,
    show_default=True,
)
@click.option(
    "--output-training-file",
    help="The path (.sqlite) to save the training set to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),

    required=True,
)
@click.option(
    "--output-test-file",
    help="The path (.sqlite) to save the test set to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-validation-file",
    help="The path (.sqlite) to save the validation set to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--element-order",
    type=str,
    multiple=True,
    help="Element order",
    default=["S", "F", "Cl", "Br", "I", "P", "O", "N"],
    show_default=True,
)
def partition_molecules_cli(
    input_file: List[str],
    training_fraction: float = 0.7,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    output_training_file: str = "training-data.sqlite",
    output_test_file: str = "test-data.sqlite",
    output_validation_file: str = "validation-data.sqlite",
    element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N"]
):
    import tqdm
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.app.partition import DatasetPartitioner

    if isinstance(element_order, str):
        element_order = [x for x in element_order.split() if x]


    if not len(input_file) >= 1:
        raise ValueError("At least one input source must be given")
    
    all_records = []
    for file in tqdm.tqdm(input_file, desc="loading from stores"):
        store = MoleculeStore(file)
        records = store.retrieve()
        all_records.extend(records)

    partitioner = DatasetPartitioner.from_molecule_records(all_records)

    all_sets = partitioner.partition(
        training_fraction=training_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        element_order=element_order,
    )
    print(f"Partitioned {len(all_records)} records into:")

    filenames = [output_training_file, output_validation_file, output_test_file]
    names = ["Training", "Validation", "Test"]

    for filename, dataset, name in zip(filenames, all_sets, names):
        store_ = MoleculeStore(filename)
        records_ = [x for x in all_records if x.mapped_smiles in dataset]
        print(f"    {name}: {len(records_)}")

        store_.store(records_)



if __name__ == "__main__":
    partition_molecules_cli()