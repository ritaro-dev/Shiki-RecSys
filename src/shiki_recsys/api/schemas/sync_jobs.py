from datetime import datetime

from pydantic import BaseModel, ConfigDict

from shiki_recsys.sync_jobs import (
    SyncJobErrorCode,
    SyncJobStatus,
)


class SyncJobResponse(BaseModel):
    """Represent a synchronization job returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    status: SyncJobStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_code: SyncJobErrorCode | None
    error_message: str | None
