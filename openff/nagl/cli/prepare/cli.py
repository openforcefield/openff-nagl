import click

from openff.nagl.cli.prepare.generate_conformers import generate_conformers_cli

@click.group(
    "prepare",
    short_help="CLIs for preparing molecule sets.",
    help="CLIs for preparing molecule sets, such as filtering out molecules which are "
    "too large or contain unwanted chemistries, removing counter-ions, or enumerating "
    "possible tautomers / protomers.",
)
def prepare_cli():
    pass


prepare_cli.add_command(generate_conformers_cli)
