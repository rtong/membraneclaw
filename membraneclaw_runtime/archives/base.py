from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ArchivePolicy:
    visibility: str = 'private'
    data_mode: str = 'private_raw'
    max_attempts: int = 3


@dataclass(frozen=True)
class ArchiveReceipt:
    status: str
    snapshot_hash: str
    branch: str | None = None
    commit: str | None = None
    error_code: str | None = None
    repository: str | None = None


class TraceArchive(Protocol):
    def publish(self, snapshot, policy: ArchivePolicy) -> ArchiveReceipt: ...
