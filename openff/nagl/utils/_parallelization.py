import contextlib
import multiprocessing
import typing

_S = typing.TypeVar("_S")
_T = typing.TypeVar("_T")


@contextlib.contextmanager
def get_mapper_to_processes(
    n_processes: int = 0,
) -> typing.Callable[
    [typing.Callable[[_S], _T], typing.Iterable[_S]], typing.Iterable[_T]
]:
    """Returns either an interactive parallel map or a standard map depending on the
    number of processes"""

    if n_processes <= 0:
        yield map
    else:
        with multiprocessing.Pool(n_processes) as pool:
            yield pool.imap
    