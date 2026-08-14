from enum import StrEnum


class SyncJobStatus(StrEnum):
    """Represent the lifecycle state of a synchronization job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncJobErrorCode(StrEnum):
    """Represent stable synchronization failure codes."""

    SYNC_FAILED = "sync_failed"
    WORKER_TIMEOUT = "worker_timeout"
