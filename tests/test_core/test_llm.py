# Copyright 2026 Zachary Brooks
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tests for LLM utility helpers."""

from safe_agent.core.llm import restore_tool_name, sanitize_tool_name


class TestSanitizeToolName:
    """Tests for sanitize_tool_name helper."""

    def test_replaces_single_colon(self) -> None:
        assert sanitize_tool_name("filesystem:read_file") == "filesystem__read_file"

    def test_replaces_only_first_colon(self) -> None:
        # SafeAgent tool names have exactly one colon; if somehow multiple
        # exist, only the first is replaced — no silent corruption downstream.
        assert sanitize_tool_name("a:b:c") == "a__b:c"

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

    def test_only_first_colon_replaced(self) -> None:
        # SafeAgent names have exactly one colon; if somehow multiple colons
        # exist, only the first is replaced — no silent corruption.
        assert sanitize_tool_name("a:b:c") == "a__b:c"

    def test_restore_only_first_double_underscore(self) -> None:
        # Symmetric: only the first __ is restored.
        assert restore_tool_name("a__b__c") == "a:b__c"
