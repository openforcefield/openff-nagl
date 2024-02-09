import contextlib
import functools
import itertools
import logging
import math
import multiprocessing
import tqdm
import typing

_S = typing.TypeVar("_S")
_T = typing.TypeVar("_T")

logger = logging.getLogger(__name__)


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


def as_batch_function(
    func: typing.Callable[[typing.Any], typing.Any],
    desc: str = "Processing",
    capture_errors: bool = False
):
    def wrapper(batch: typing.Iterable[typing.Any], *args, **kwargs):
        results = [
            func(entry, *args, **kwargs)
            for entry in tqdm.tqdm(batch, ncols=80, desc=desc)
        ]
        return results

    def error_wrapper(batch: typing.Iterable[typing.Any], *args, **kwargs):
        results = []
        for entry in tqdm.tqdm(batch, ncols=80, desc=desc):
            error = None
            try:
                result = func(entry, *args, **kwargs)
            except BaseException as e:
                result = None
                error = f"Failed to process {entry}: {e}"
            results.append((result, error))
        return results

    if capture_errors:
        return error_wrapper
    return wrapper


def batch_entries(
    entries: typing.Iterable[_S],
    batch_size: int = 1,
):
    size = batch_size - 1
    entries = iter(entries)
    for x in entries:
        yield list(itertools.chain([x], itertools.islice(entries, size)))


def reconcile_batch_workers(
    entries: typing.Iterable[_S],
    n_entries: typing.Optional[int] = None,
    batch_size: int = -1,
    n_workers: int = -1,
):
    if n_entries is None:
        n_entries = len(entries)
    if batch_size < 0:
        n_batches = 1
        batch_size = n_entries
    else:
        n_batches = int(math.ceil(n_entries / batch_size))

    if n_workers < 0:
        n_workers = n_batches
    if n_workers > n_batches:
        n_workers = n_batches

    return n_workers, n_batches, batch_size


@contextlib.contextmanager
def batch_distributed(
    entries: typing.Iterable[typing.Any],
    n_entries: typing.Optional[int] = None,
    batch_size: int = -1,
    n_workers: int = -1,
    worker_type: typing.Literal["lsf", "slurm", "local"] = "local",
    queue: str = "cpuqueue",
    account: typing.Optional[str] = None,
    conda_environment: str = "openff-nagl",
    memory: int = 4,  # GB
    walltime: int = 32,  # hours
    package_manager: typing.Literal["conda", "micromamba"] = "conda",
    **kwargs
):
    import dask
    from distributed import LocalCluster
    from dask import distributed
    from dask_jobqueue import LSFCluster, SLURMCluster

    n_workers, n_batches, batch_size = reconcile_batch_workers(
        entries, n_entries, batch_size, n_workers
    )

    logger.warning(
        f"Setting n_workers={n_workers} for {n_batches} batches"
    )

    env_extra = []
    if worker_type != "local":
        env_extra.extend(
            dask.config.get(f"jobqueue.{worker_type}.job-script-prologue", default=[])
        )
    env_extra.append(f"{package_manager} activate {conda_environment}")


    if worker_type == "local":
        cluster = LocalCluster(n_workers=n_workers)
    else:
        CLUSTER_CLASSES = {
            "lsf": LSFCluster,
            "slurm": SLURMCluster,
        }
        cluster = CLUSTER_CLASSES[worker_type](
            queue=queue,
            project=account,
            cores=1,
            memory=f"{memory * 1e9}B",
            walltime=f"{walltime}:00",
            job_script_prologue=env_extra,
        )
        cluster.scale(n=n_workers)
    
    client = distributed.Client(cluster)

    def wrapper(func, **kwargs):
        for batch in batch_entries(entries, batch_size):
            future = client.submit(func, batch, **kwargs)
            yield future

    try:
        yield wrapper
    finally:
        if worker_type != "local":
            cluster.scale(n=0)