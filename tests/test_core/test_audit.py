"""Tests for safe_agent.core.audit — AuditLogger and AuditEntry."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from safe_agent.core.audit import AuditEntry, AuditLogger
from safe_agent.iam.models import Decision


def _make_entry(**overrides) -> AuditEntry:
    """Create a minimal AuditEntry for testing."""
    defaults = {
        "session_id": "sess-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "tool_name": "fs:ReadFile",
        "params": {"path": "/etc/hosts"},
        "resolved_conditions": {"env": "prod"},
        "decision": Decision.ALLOWED,
        "matched_statements": ["AllowRead"],
    }
    defaults.update(overrides)
    return AuditEntry(**defaults)


class TestAuditEntry:
    """Tests for AuditEntry model."""

    def test_required_fields(self) -> None:
        """AuditEntry should store all required fields."""
        entry = _make_entry()
        assert entry.session_id == "sess-1"
        assert entry.tool_name == "fs:ReadFile"
        assert entry.decision == Decision.ALLOWED

    def test_decision_must_be_decision_enum(self) -> None:
        """AuditEntry.decision must be a Decision enum value."""
        entry = _make_entry(decision=Decision.DENIED_EXPLICIT)
        assert entry.decision == Decision.DENIED_EXPLICIT

    def test_defaults(self) -> None:
        """Optional fields default to empty collections."""
        entry = AuditEntry(
            session_id="s",
            timestamp="2026-01-01T00:00:00+00:00",
            tool_name="t",
            decision=Decision.DENIED_IMPLICIT,
        )
        assert entry.params == {}
        assert entry.resolved_conditions == {}
        assert entry.matched_statements == []

    def test_serialises_to_json(self) -> None:
        """AuditEntry.model_dump_json() should produce valid JSON."""
        entry = _make_entry()
        raw = entry.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["session_id"] == "sess-1"
        assert parsed["decision"] == "ALLOWED"

    def test_matched_statements_allows_none(self) -> None:
        """matched_statements may contain None for anonymous statements."""
        entry = _make_entry(matched_statements=[None, "SomePolicy"])
        assert entry.matched_statements == [None, "SomePolicy"]

    def test_invalid_timestamp_raises(self) -> None:
        """AuditEntry should reject timestamps that are not valid ISO-8601."""
        with pytest.raises(Exception, match="ISO-8601"):
            _make_entry(timestamp="not-a-timestamp")

    def test_valid_timestamp_accepted(self) -> None:
        """AuditEntry should accept well-formed ISO-8601 timestamps."""
        entry = _make_entry(timestamp="2026-06-01T12:00:00+00:00")
        assert entry.timestamp == "2026-06-01T12:00:00+00:00"


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_creates_file_on_first_log(self, tmp_path: Path) -> None:
        """log() should create the log file if it does not exist."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        assert not log_file.exists()
        logger.log(_make_entry())
        assert log_file.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """AuditLogger.__init__ should create missing parent directories."""
        log_file = tmp_path / "deep" / "nested" / "audit.jsonl"
        AuditLogger(log_path=log_file)
        assert log_file.parent.exists()

    def test_entry_written_as_json_line(self, tmp_path: Path) -> None:
        """Each logged entry should be a valid JSON line."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry())
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tool_name"] == "fs:ReadFile"

    def test_multiple_entries_appended(self, tmp_path: Path) -> None:
        """Multiple log() calls should append separate JSON lines."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry(decision=Decision.ALLOWED))
        logger.log(_make_entry(decision=Decision.DENIED_EXPLICIT))
        logger.log(_make_entry(decision=Decision.DENIED_IMPLICIT))
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3
        decisions = [json.loads(ln)["decision"] for ln in lines]
        assert decisions == ["ALLOWED", "DENIED_EXPLICIT", "DENIED_IMPLICIT"]

    def test_entry_contains_all_required_fields(self, tmp_path: Path) -> None:
        """Each written entry must contain all required AuditEntry fields."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry())
        parsed = json.loads(log_file.read_text().strip())
        for field in (
            "session_id",
            "timestamp",
            "tool_name",
            "params",
            "resolved_conditions",
            "decision",
            "matched_statements",
        ):
            assert field in parsed, f"Missing field: {field}"

    def test_read_entries_empty_when_no_file(self, tmp_path: Path) -> None:
        """read_entries() should return [] when the log file does not exist."""
        logger = AuditLogger(log_path=tmp_path / "sub" / "missing.jsonl")
        assert logger.read_entries() == []

    def test_read_entries_roundtrip(self, tmp_path: Path) -> None:
        """read_entries() should return AuditEntry objects matching what was logged."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        entry = _make_entry()
        logger.log(entry)
        entries = logger.read_entries()
        assert len(entries) == 1
        assert entries[0].session_id == entry.session_id
        assert entries[0].decision == entry.decision

    def test_read_entries_limit(self, tmp_path: Path) -> None:
        """read_entries(limit=N) should return at most N entries."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        for _ in range(5):
            logger.log(_make_entry())
        assert len(logger.read_entries(limit=3)) == 3
        assert len(logger.read_entries(limit=10)) == 5

    def test_now_iso_returns_utc_string(self) -> None:
        """now_iso() should return a valid ISO-8601 UTC string."""
        from datetime import datetime, timedelta

        ts = AuditLogger.now_iso()
        assert isinstance(ts, str)
        # Must parse as a valid ISO-8601 datetime
        parsed = datetime.fromisoformat(ts)
        # Must be UTC (offset zero)
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_iter_entries_skips_malformed_lines(
        self, tmp_path: Path, caplog
    ) -> None:
        """iter_entries() should skip malformed lines and emit a warning."""
        import logging

        log_file = tmp_path / "audit.jsonl"
        logger_inst = AuditLogger(log_path=log_file)
        # Write one good entry, one bad line, one good entry
        logger_inst.log(_make_entry(decision=Decision.ALLOWED))
        log_file.open("a").write("NOT JSON AT ALL\n")
        logger_inst.log(_make_entry(decision=Decision.DENIED_EXPLICIT))

        with caplog.at_level(logging.WARNING, logger="safe_agent.core.audit"):
            entries = logger_inst.read_entries()

        assert len(entries) == 2
        assert entries[0].decision == Decision.ALLOWED
        assert entries[1].decision == Decision.DENIED_EXPLICIT
        assert any("skipping unparseable" in r.message for r in caplog.records)

    def test_concurrent_writes_no_interleaving(self, tmp_path: Path) -> None:
        """Concurrent log() calls from multiple threads must not interleave lines."""
        log_file = tmp_path / "audit.jsonl"
        logger_inst = AuditLogger(log_path=log_file)
        n_threads = 20
        n_per_thread = 50

        def write_entries(thread_id: int) -> None:
            for i in range(n_per_thread):
                logger_inst.log(
                    _make_entry(
                        session_id=f"thread-{thread_id}",
                        tool_name=f"tool:{thread_id}:{i}",
                    )
                )

        threads = [
            threading.Thread(target=write_entries, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = logger_inst.read_entries()
        assert len(entries) == n_threads * n_per_thread
        # Every line must be valid JSON (no interleaving partial writes)
        raw_lines = log_file.read_text().strip().splitlines()
        for line in raw_lines:
            json.loads(line)  # raises on corruption
