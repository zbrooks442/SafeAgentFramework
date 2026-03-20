"""Audit logger for SafeAgent tool dispatch decisions."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from safe_agent.iam.models import Decision

logger = logging.getLogger(__name__)

# Maximum serialised byte length for params/resolved_conditions stored in an
# audit entry. Values exceeding this limit are replaced with a truncation
# marker so the audit log cannot become a credential or bulk-data dump.
_MAX_PARAM_BYTES = 8 * 1024  # 8 KB
_TRUNCATION_MARKER = "__truncated__"


def _truncate_params(value: dict[str, Any]) -> dict[str, Any]:
    """Return *value* unchanged, or a truncation marker if it is too large.

    The entire dict is replaced (not partially redacted) to avoid creating
    records that appear complete but are missing fields.

    Args:
        value: The params or conditions dict to check.

    Returns:
        The original dict, or ``{"__truncated__": True}`` if it serialises
        to more than ``_MAX_PARAM_BYTES`` bytes.
    """
    try:
        encoded = json.dumps(value, default=str)
    except Exception:
        return {_TRUNCATION_MARKER: True}
    if len(encoded.encode()) > _MAX_PARAM_BYTES:
        return {_TRUNCATION_MARKER: True}
    return value


class AuditEntry(BaseModel):
    """A structured record of a single authorization decision.

    Attributes:
        session_id: Identifier for the session that triggered the tool call.
        timestamp: ISO-8601 UTC timestamp of the decision. Must be a valid
            ISO-8601 string (validated at construction time).
        tool_name: The fully-qualified tool name that was evaluated.
        params: Input parameters for the tool call, capped at
            ``_MAX_PARAM_BYTES`` (8 KB). Values exceeding the cap are replaced
            with ``{"__truncated__": True}``. Callers should also redact
            known-sensitive fields (API keys, passwords, PII) before
            constructing the entry.
        resolved_conditions: Condition values resolved by the module; also
            size-capped.
        decision: The authorization decision.
        matched_statements: Sids of policy statements that contributed to the
            decision. ``None`` entries represent anonymous (no-sid) statements.
            The sentinel values ``"__unknown_tool__"``, ``"__internal_error__"``,
            ``"__execute_error__"``, and ``"__executed__"`` are used by the
            dispatcher for machine-readable event classification.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: str
    timestamp: str
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    resolved_conditions: dict[str, Any] = Field(default_factory=dict)
    decision: Decision
    matched_statements: list[str | None] = Field(default_factory=list)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, v: Any) -> Any:
        """Reject timestamps that are not valid ISO-8601 strings."""
        if isinstance(v, str):
            try:
                datetime.fromisoformat(v)
            except ValueError as exc:
                raise ValueError(
                    f"timestamp must be a valid ISO-8601 string, got: {v!r}"
                ) from exc
        return v

    @field_validator("params", "resolved_conditions", mode="before")
    @classmethod
    def _cap_size(cls, v: Any) -> Any:
        """Enforce the per-field byte cap before storing."""
        if isinstance(v, dict):
            return _truncate_params(v)
        return v


class AuditLogger:
    """Appends structured :class:`AuditEntry` records as JSON lines to a file.

    **Write path** — :meth:`log` is fully thread-safe: a :class:`threading.Lock`
    serialises concurrent appends so that log lines are never interleaved.

    **Read path** — :meth:`iter_entries` and :meth:`read_entries` intentionally
    do **not** hold the write lock while reading the file. Because :meth:`log`
    only ever appends (never seeks or truncates), a concurrent read at worst
    observes a partial final line, which the malformed-line skip path handles
    gracefully. Holding the write lock during a potentially large file read
    would stall every concurrent dispatch call — unacceptable for the
    enforcement gate.

    **Multi-process safety** — ``threading.Lock`` only serialises threads
    within the same process. If multiple processes write to the same log file,
    external file locking (e.g. ``fcntl.flock``) is required; this class does
    not provide it.

    Args:
        log_path: Path to the JSON-lines audit log file.

    Example::

        logger = AuditLogger(log_path=Path("/var/log/safe_agent/audit.jsonl"))
        logger.log(entry)
    """

    def __init__(self, log_path: Path) -> None:
        """Initialise the audit logger.

        Args:
            log_path: Destination file for JSON-lines audit records. Parent
                directories are created eagerly so that logging never fails
                due to a missing directory.
        """
        self._log_path = log_path
        self._lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string.

        Intended for use by callers (e.g. the dispatcher) that need to stamp
        a logical event time before constructing :class:`AuditEntry` objects.
        """
        return datetime.now(tz=UTC).isoformat()

    def log(self, entry: AuditEntry) -> None:
        """Append *entry* as a JSON line to the audit log file.

        Thread-safe within a single process. The lock ensures no two concurrent
        callers interleave partial writes.

        Args:
            entry: The :class:`AuditEntry` to persist.
        """
        line = entry.model_dump_json() + "\n"
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def read_entries(self, limit: int | None = None) -> list[AuditEntry]:
        """Read and parse entries from the audit log.

        The file is read and closed before any entries are returned, so no file
        descriptor is held open. Safe to call concurrently with :meth:`log`.

        Note: the entire file is read into memory regardless of *limit*; for
        very large audit files prefer :meth:`iter_entries` with a limit.

        Args:
            limit: If provided, return at most *limit* entries from the start
                of the file. ``None`` returns all entries.

        Returns:
            A list of :class:`AuditEntry` objects in file order. Returns an
            empty list if the log file does not yet exist.
        """
        return list(self.iter_entries(limit=limit))

    def iter_entries(self, limit: int | None = None) -> Iterator[AuditEntry]:
        """Iterate over entries from the audit log.

        The file is read fully into a string before parsing begins, so no file
        descriptor is held open across ``yield`` points.

        The write lock is intentionally **not** held during the read. See class
        docstring for the rationale and trade-offs.

        Args:
            limit: Stop after yielding *limit* entries. ``None`` means all.

        Yields:
            :class:`AuditEntry` objects in file order. Malformed lines are
            skipped with a ``WARNING`` log and do not raise.
        """
        try:
            raw_content = self._log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return

        count = 0
        for raw in raw_content.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                yield AuditEntry(**json.loads(line))
            except Exception:
                logger.warning(
                    "audit: skipping unparseable log line: %.120r", line
                )
                continue
            count += 1
            if limit is not None and count >= limit:
                return
