import click


@click.command("similarity")
@click.option(
    "--input-file",
    help="The path to the input SQLITE store file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-file",
    help="The path to the SDF file (.sdf) to save the generated conformers in.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
def plot_similarity_cli(input_file, output_file, linecolor="white"):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    df = pd.read_csv(input_file)
    smiles = df.smiles_1.unique()
    n_smiles = len(smiles)
    data = np.zeros((n_smiles, n_smiles))
    iu = np.triu_indices(n_smiles)
    data[iu] = df.similarity.values
    data += data.T

    # wide = df.pivot(
    #     index="smiles_1",
    #     columns="smiles_2",
    #     values="similarity"
    # )
    # print(wide)

    smiles_to_source = {}
    for row in df.itertuples():
        smiles_to_source[row.smiles_1] = row.source_1

    original_ticklabels = [smiles_to_source[smi] for smi in smiles]
    df_labels = pd.DataFrame({"source": original_ticklabels})
    tick_bounds = {}
    i = 0
    for k, v in df_labels.groupby("source"):
        j = i + len(v)
        tick_bounds[k] = (i, j)
        i = j
    ticklabels = [""] * len(original_ticklabels)
    for label, (i, j) in tick_bounds.items():
        midpoint = int((i + j) / 2)
        ticklabels[midpoint] = label

    ax = sns.heatmap(
        # data=wide,
        data=data,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=False,
        square=True,
        cbar=True,
        cbar_kws={"label": "Similarity"},
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    )

    for i, j in tick_bounds.values():
        plt.axhline(y=i, c=linecolor)
        plt.axhline(y=j, c=linecolor)
        plt.axvline(x=i, c=linecolor)
        plt.axvline(x=j, c=linecolor)

    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    plot_similarity_cli()
