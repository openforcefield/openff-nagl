import click

from openff.nagl.cli.plot.plot_similarity import plot_similarity_cli

__all__ = [
    "plot_cli",
]


@click.group(
    "plot",
    short_help="CLIs for plotting.",
    help="CLIs for plotting.",
)
def plot_cli():
    pass


plot_cli.add_command(plot_similarity_cli)
