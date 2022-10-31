import json
from typing import List
import pathlib


import click

@click.command(
    "partition",
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
    "--input-source-file",
    help="The path to the input source information (JSON) for data",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    default=None,
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
    "--output-source-file",
    help="The path (.csv) to save the source information for data.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="partitioned-data-sources.csv",
    show_default=True,
)
@click.option(
    "--clean-filenames",
    help="If on, only save the base filename instead of the whole path",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--seed",
    help="Seed for diverse selection",
    type=int,
    default=-1,
    show_default=True,
)
def partition_molecules_cli(
    input_file: List[str],
    input_source_file: str = None,
    training_fraction: float = 0.7,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    output_training_file: str = "training-data.sqlite",
    output_test_file: str = "test-data.sqlite",
    output_validation_file: str = "validation-data.sqlite",
    output_source_file: str = "partitioned-data-sources.csv",
    clean_filenames: bool = True,
    seed: int = -1
):
    import pandas as pd
    import tqdm
    from openff.nagl.storage.store import MoleculeStore
    from openff.nagl.app.partitioner import DatasetPartitioner

    if not len(input_file) >= 1:
        raise ValueError("At least one input source must be given")

    input_source = None
    if input_source_file is not None:
        with open(input_source_file, "r") as f:
            input_source = json.load(f)
    
    smiles_to_records = {}
    smiles_to_db = {}
    for file in tqdm.tqdm(input_file, desc="loading from stores"):
        store = MoleculeStore(file)
        records = store.retrieve()
        if clean_filenames:
            file = pathlib.Path(file).stem
        for rec in records:
            smiles_to_records[rec.mapped_smiles] = rec
            if input_source is not None:
                smiles_to_db[rec.mapped_smiles] = input_source.get(rec.mapped_smiles)
            else:
                smiles_to_db[rec.mapped_smiles] = file

    smiles_to_db = {
        k: v if v is not None else ""
        for k, v in smiles_to_db.items()
    }

    partitioner = DatasetPartitioner(smiles_to_db)
    all_sets = partitioner.partition(
        training_fraction=training_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed
    )

    print(f"Partitioned {len(smiles_to_records)} records into:")

    filenames = [output_training_file, output_validation_file, output_test_file]
    names = ["Training", "Validation", "Test"]
    dfs = []

    for filename, dataset, name in zip(filenames, all_sets, names):
        store_ = MoleculeStore(filename)
        records_ = [smiles_to_records[x] for x in dataset.labelled_smiles]
        if len(records_):
            smiles, sources = zip(*(dataset.labelled_smiles.items()))
            df_ = pd.DataFrame({
                "SMILES": smiles,
                "Source": sources
            })
            df_["Dataset"] = name
            dfs.append(df_)
        print(f"    {name}: {len(records_)}")

        store_.store(records_)
    
    df = pd.concat(dfs).reset_index()
    df.to_csv(output_source_file)



if __name__ == "__main__":
    partition_molecules_cli()