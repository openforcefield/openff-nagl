import logging
import typing

from openff.utilities import requires_package

if typing.TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

@requires_package("pyarrow")
def _append_column_to_table(
    table: "pa.Table",
    key: typing.Union["pa.Field", str],
    values: typing.Iterable[typing.Any],
    exist_ok: bool = False
):
    import pyarrow as pa
    if isinstance(key, pa.Field):
        k_name = key.name
    else:
        k_name = key
    if k_name in table.column_names:
        if exist_ok:
            logger.warning(
                f"Column {k_name} already exists in table. "
                "Overwriting."
            )
            table = table.drop_columns(k_name)
        else:
            raise ValueError(f"Column {k_name} already exists in table")

    table = table.append_column(key, [values])
    return table