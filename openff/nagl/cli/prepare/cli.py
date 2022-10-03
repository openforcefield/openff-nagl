import click

from openff.nagl.cli.prepare.generate_conformers import generate_conformers_cli
from openff.nagl.cli.prepare.select import select_molecules_cli
from openff.nagl.cli.prepare.partition import partition_molecules_cli

@click.group(
    "prepare",
    short_help="CLIs for selecting molecule sets.",
    help="CLIs for preparing molecule sets, such as filtering out molecules which are "
    "too large or contain unwanted chemistries, removing counter-ions, or enumerating "
    "possible tautomers / protomers.",
)
def prepare_cli():
    pass


prepare_cli.add_command(generate_conformers_cli)
prepare_cli.add_command(select_molecules_cli)
prepare_cli.add_command(partition_molecules_cli)
