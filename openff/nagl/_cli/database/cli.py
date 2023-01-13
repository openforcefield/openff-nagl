import click

from openff.nagl._cli.database.retrieve import retrieve_molecules_cli
from openff.nagl._cli.database.store import store_molecules_cli


@click.group(
    "database",
    short_help="CLIs for interacting with databases.",
    help="CLIs for interacting with databases, such as storing and retrieving molecules.",
)
def database_cli():
    pass


database_cli.add_command(store_molecules_cli)
database_cli.add_command(retrieve_molecules_cli)
