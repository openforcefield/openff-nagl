import dataclasses
import functools
import logging
import math
from typing import Any, Callable, List, Literal, NamedTuple, Optional

logger = logging.getLogger(__name__)


class BatchSettings(NamedTuple):
    batch_size: int
    n_workers: int = -1
    n_batches: int = -1

    def batch_entries(self, entries: List[Any]) -> List[List[Any]]:
        import itertools

        size = self.batch_size - 1
        entries = iter(entries)
        for x in entries:
            yield list(itertools.chain([x], itertools.islice(entries, size)))

@dataclasses.dataclass
class ClusterManager:
    batch_size: int = -1
    n_workers: int = -1
    worker_type: Literal["lsf", "local", "slurm"] = "local"
    queue: str = "cpuqueue"
    conda_environment: str = "openff-nagl"
    memory: int = 4  # GB
    walltime: int = 32  # hours

    def reconcile_batch_workers(self, n_entries: int) -> BatchSettings:
        batch_size = self.batch_size
        n_workers = self.n_workers
        n_batches = 1

        if batch_size < 0:
            batch_size = n_entries
        else:
            n_batches = int(math.ceil(n_entries / batch_size))
        
        if n_workers < 0:
            n_workers = n_batches
        elif n_workers > n_batches:
            n_workers = n_batches
            logger.warning(
                f"More workers ({n_workers}) requested "
                f"than batches to compute ({n_batches}). "
                f"Setting n_workers={n_batches}"
            )

        return BatchSettings(batch_size, n_workers, n_batches)

    
    def _launch_hpc_cluster(self):
        import dask
        from dask_jobqueue import LSFCluster, SLURMCluster

        cluster_type = {
            "lsf": LSFCluster,
            "slurm": SLURMCluster,
        }

        worker_type = self.worker_type.lower()
        try:
            cluster_class = cluster_type[worker_type]
        except KeyError:
            raise ValueError(
                f"Unknown worker type {self.worker_type}. "
                f"Valid types are {list(cluster_type.keys())}"
            )

        env_extra = dask.config.get(
            f"jobqueue.{worker_type}.job-script-prologue",
            default=[]
        )
        if not env_extra:
            env_extra = []
        env_extra.append(f"conda activate {self.conda_environment}")

        cluster = cluster_class(
            queue=self.queue,
            cores=1,
            memory=f"{self.memory}GB",
            walltime=f"{self.walltime}:00",
            local_directory="dask-worker-space",
            log_directory="dask-worker-logs",
            job_script_prologue=env_extra,
        )
        return cluster

    def _launch_cluster(self, batch_settings: Optional[BatchSettings] = None):
        from distributed import LocalCluster

        if batch_settings is None:
            n_workers = max(self.n_workers, 1)
        else:
            n_workers = batch_settings.n_workers

        if self.worker_type == "local":
            cluster = LocalCluster(n_workers=n_workers)
        else:
            cluster = self._launch_hpc_cluster()
            cluster.scale(n_workers)
        return cluster