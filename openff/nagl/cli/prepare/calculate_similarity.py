import functools
import itertools
import math
import pathlib
from typing import Dict, Tuple, Union

import click
import tqdm


def calculate_similarity(
    smiles_pair: Tuple[str, str],
    radius: int = 3,
) -> float:
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
    from rdkit.DataStructs import DiceSimilarity

    ref_smiles, target_smiles = smiles_pair
    ref = Chem.MolFromSmiles(ref_smiles)
    ref_fp = GetMorganFingerprint(ref, radius)
    target = Chem.MolFromSmiles(target_smiles)
    target_fp = GetMorganFingerprint(target, radius)

    return DiceSimilarity(ref_fp, target_fp)


def get_similarity(
    smiles_pair: Tuple[str, str],
    radius: int = 3,
) -> Dict[str, Union[str, float]]:
    ref_smiles, target_smiles = smiles_pair
    similarity = calculate_similarity(smiles_pair, radius)
    return {"smiles_1": ref_smiles, "smiles_2": target_smiles, "similarity": similarity}



def calculate_all_similarity(
    input_files,
    output_file,
    manager,
    clean_filenames: bool = True,
    fingerprint_radius: int = 3,
    skip: int = 10,
):
    import numpy as np
    import pandas as pd

    from openff.nagl.cli.utils import (
        as_batch_function_with_captured_errors,
        preprocess_args,
    )
    from openff.nagl.storage.store import MoleculeStore

    # file_order = {}
    all_smiles = {}
    for i, file in tqdm.tqdm(enumerate(input_files), desc="Loading from stores"):
        # file_order[file] = i
        smiles = sorted(MoleculeStore(file).get_smiles())[::skip]

        if clean_filenames:
            file = str(pathlib.Path(file).stem)
        file_smiles = dict.fromkeys(smiles, file)
        all_smiles.update(file_smiles)
    pairwise_smiles = itertools.combinations(all_smiles, 2)
    n_smiles = len(all_smiles)
    n_pairs = int(math.factorial(n_smiles) / (math.factorial(n_smiles - 2) * 2))
    results = np.zeros((n_pairs,), dtype=np.float16)
    print(results.nbytes)

    print(f"Calculating {n_pairs} pairs of similarity")

    manager, *_ = preprocess_args(manager=manager)
    manager.set_entries(pairwise_smiles, n_entries=n_pairs)

    single_func = functools.partial(get_similarity, radius=fingerprint_radius)
    batch_func = as_batch_function_with_captured_errors(
        single_func, desc="calculating similarity"
    )

    with manager:
        futures = manager.submit_to_client(batch_func)
        i = 0
        for future in tqdm.tqdm(futures, desc="gathering similarity"):
            for similarity, error in future.result():
                results[i] = similarity["similarity"]
                i += 1
            # results.extend(future.result())

    df = pd.DataFrame({"similarity": results})
    smiles_1, smiles_2 = zip(*itertools.combinations(all_smiles, 2))
    df["smiles_1"] = smiles_1
    df["smiles_2"] = smiles_2

    # results = [x[0] for x in results]
    # df = pd.DataFrame.from_records(results)
    df["source_1"] = [all_smiles[x] for x in df.smiles_1]
    df["source_2"] = [all_smiles[x] for x in df.smiles_2]
    df.to_csv(output_file)
    print(f"Saved dataframe to {output_file}")


@click.command("calculate-similarity", help="Calculate similarity between datasets")
@click.option(
    "--input-file",
    help="The path to the input SQLITE store file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option(
    "--output-file",
    help="The path to the SDF file (.sdf) to save the generated conformers in.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--clean-filenames",
    help="If on, only save the base filename instead of the whole path",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--fingerprint-radius",
    help="Fingerprint radius",
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    "--skip",
    help="Include every `skip`th molecule from each file",
    type=int,
    default=10,
    show_default=True,
)
@click.pass_context
def calculate_similarity_cli(
    ctx, input_file, output_file, clean_filenames, fingerprint_radius, skip
):
    from openff.nagl.cli.utils import get_default_manager

    manager = get_default_manager(ctx)
    calculate_all_similarity(
        input_file, output_file, manager, clean_filenames, fingerprint_radius, skip
    )


if __name__ == "__main__":
    calculate_similarity_cli()
