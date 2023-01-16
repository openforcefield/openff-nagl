import contextlib
import dataclasses
import functools
import logging
import math
import traceback
from typing import Any, Callable, List, Literal

import tqdm

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Manager:
    """Helper class to manage batched work on a cluster"""

    batch_size: int = -1
    n_workers: int = -1
    worker_type: Literal["lsf", "local"] = "local"
    queue: str = "cpuqueue"
    conda_environment: str = "openff-nagl"
    memory: int = 4  # GB
    walltime: int = 32  # hours

    def __post_init__(self):
        self.entries = []
        self.n_entries = 0
        self.cluster = None
        self.client = None

    def set_entries(self, entries: List[Any], n_entries=None):
        self.entries = entries
        if n_entries is None:
            n_entries = len(entries)
        self.n_entries = n_entries
        self.reconcile_batch_workers()

    def reconcile_batch_workers(self):
        if self.batch_size < 0:
            n_batches = 1
            self.batch_size = self.n_entries
        else:
            n_batches = int(math.ceil(self.n_entries / self.batch_size))
        n_workers = self.n_workers
        if n_workers < 0:
            n_workers = n_batches
        if n_workers > n_batches:
            n_workers = n_batches
            logger.warning(
                f"More workers ({n_workers}) requested "
                f"than batches to compute ({n_batches}). "
                f"Setting n_workers={n_batches}"
            )
        self.n_batches = n_batches
        self.n_workers = n_workers

    def batch_entries(self):
        import itertools
        # contort around generators
        size = self.batch_size - 1
        entries = iter(self.entries)
        for x in entries:
            yield list(itertools.chain([x], itertools.islice(entries, size)))

        # for i in range(0, self.n_entries, self.batch_size):
        #     j = min(i + self.batch_size, self.n_entries)
        #     yield self.entries[i:j]

    def setup_lsf_cluster(self):
        import dask
        from dask_jobqueue import LSFCluster

        env_extra = dask.config.get("jobqueue.lsf.job-script-prologue", default=[])
        if not env_extra:
            env_extra = []
        env_extra.append(f"conda activate {self.conda_environment}")

        cluster = LSFCluster(
            queue=self.queue,
            cores=1,
            memory=f"{self.memory * 1e9}B",
            walltime=f"{self.walltime}:00",
            local_directory="dask-worker-space",
            log_directory="dask-worker-logs",
            job_script_prologue=env_extra,
        )
        cluster.scale(n=self.n_workers)
        return cluster

    def _set_up_cluster(self):
        from distributed import LocalCluster

        if self.worker_type == "lsf":
            cluster = self.setup_lsf_cluster()
        elif self.worker_type == "local":
            cluster = LocalCluster(n_workers=self.n_workers)
        else:
            raise NotImplementedError()
        return cluster

    def set_up_cluster(self):
        from dask import distributed

        if self.cluster is None:
            self.cluster = self._set_up_cluster()
            self.client = distributed.Client(self.cluster)

    def submit_to_client(self, submit_function, *args, **kwargs):
        self.set_up_cluster()
        futures = [
            self.client.submit(submit_function, batch, *args, **kwargs) for batch in self.batch_entries()
        ]
        return futures

    def __enter__(self):
        self.set_up_cluster()
        return self

    def __exit__(self, *args):
        self.conclude()

    def conclude(self):
        if self.worker_type == "lsf":
            self.cluster.scale(n=0)
        # self.client.shutdown()
        # self.cluster = self.client = None

    @staticmethod
    def store_futures_and_log(
        futures,
        store_function: Callable,
        log_file: str,
        aggregate_function: Callable = lambda x: x,
        n_batches: int = None,
        desc: str = None,
    ):
        from dask import distributed

        from openff.nagl._cli.utils import (
            try_and_return_error,
            write_error_to_file_object,
        )

        log_file = str(log_file)
        with open(log_file, "w") as f:
            for future in tqdm.tqdm(
                distributed.as_completed(futures, raise_errors=False),
                total=n_batches,
                desc=desc,
                ncols=80,
            ):

                def aggregator():
                    return aggregate_function(future.result())

                results, error = try_and_return_error(aggregator)
                if error is not None:
                    write_error_to_file_object(f, error)
                    continue

                for result, error in tqdm.tqdm(
                    results,
                    desc="storing batch",
                    ncols=80,
                ):
                    if result is not None and error is None:
                        storer = functools.partial(store_function, result)
                        _, error = try_and_return_error(
                            storer,
                            error="Could not store result",
                        )

                    if error is not None:
                        write_error_to_file_object(f, error)

                future.release()

        logger.info(f"Logged errors to {log_file}")
