"""Tests for LLM utility helpers."""

from safe_agent.core.llm import restore_tool_name, sanitize_tool_name


class TestSanitizeToolName:
    """Tests for sanitize_tool_name helper."""

    def test_replaces_single_colon(self) -> None:
        assert sanitize_tool_name("filesystem:read_file") == "filesystem__read_file"

    def test_replaces_multiple_colons(self) -> None:
        assert sanitize_tool_name("a:b:c") == "a__b__c"

    def test_no_colon_unchanged(self) -> None:
        assert sanitize_tool_name("no_colon_here") == "no_colon_here"

    def test_empty_string(self) -> None:
        assert sanitize_tool_name("") == ""

    def test_colon_only(self) -> None:
        assert sanitize_tool_name(":") == "__"


class TestRestoreToolName:
    """Tests for restore_tool_name helper."""

    def test_restores_double_underscore(self) -> None:
        assert restore_tool_name("filesystem__read_file") == "filesystem:read_file"

    def test_restores_multiple(self) -> None:
        assert restore_tool_name("a__b__c") == "a:b:c"

    def test_no_double_underscore_unchanged(self) -> None:
        assert restore_tool_name("no_separator") == "no_separator"

    def test_empty_string(self) -> None:
        assert restore_tool_name("") == ""


class TestRoundTrip:
    """Tests for sanitize/restore round-trip."""

    def test_sanitize_then_restore(self) -> None:
        original = "filesystem:read_file"
        assert restore_tool_name(sanitize_tool_name(original)) == original

    def test_restore_then_sanitize(self) -> None:
        sanitized = "shell__run_command"
        assert sanitize_tool_name(restore_tool_name(sanitized)) == sanitized
