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

"""Tests for the database module."""

from typing import Any
from unittest.mock import AsyncMock

from safe_agent.modules.database import DatabaseModule


class MockDatabaseBackend:
    """Mock implementation of DatabaseBackend for testing."""

    def __init__(self) -> None:
        """Initialize the mock backend with async mocks."""
        self.query = AsyncMock(return_value={"rows": [], "row_count": 0})
        self.execute_statement = AsyncMock(return_value={"rows_affected": 0})

    async def query_method(
        self,
        database: str,
        sql: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate to the mock query method."""
        return await self.query(database, sql, **kwargs)

    async def execute_statement_method(
        self,
        database: str,
        sql: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate to the mock execute_statement method."""
        return await self.execute_statement(database, sql, **kwargs)


class TestDatabaseModule:
    """Tests for DatabaseModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "database"
        assert len(descriptor.tools) == 2
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "database:query",
            "database:execute_statement",
        }

    def test_describe_has_correct_tool_names(self) -> None:
        """Tool names should follow namespace:snake_case format."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.name.startswith("database:")
            # Check snake_case format (no PascalCase)
            suffix = tool.name.removeprefix("database:")
            assert suffix.islower() or "_" in suffix

    def test_describe_has_correct_actions(self) -> None:
        """Actions should follow namespace:PascalCase format."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()

        actions = {tool.action for tool in descriptor.tools}
        assert actions == {
            "database:Query",
            "database:ExecuteStatement",
        }

    def test_describe_has_correct_resource_params(self) -> None:
        """Each tool should have 'database' as resource_param."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.resource_param == ["database"]

    def test_describe_has_correct_condition_keys(self) -> None:
        """Tools should declare database:DatabaseName and database:TableName."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert "database:DatabaseName" in tool.condition_keys
            assert "database:TableName" in tool.condition_keys

    def test_describe_parameters_have_additional_properties_false(self) -> None:
        """Parameter schemas should have additionalProperties: False."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.parameters.get("additionalProperties") is False

    async def test_execute_delegates_query_to_backend(self) -> None:
        """execute() should delegate query to the backend."""
        backend = MockDatabaseBackend()
        backend.query.return_value = {
            "rows": [{"id": 1, "name": "Alice"}],
            "row_count": 1,
        }
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "mydb", "sql": "SELECT * FROM users"},
        )

        assert result.success is True
        assert result.data == {"rows": [{"id": 1, "name": "Alice"}], "row_count": 1}
        backend.query.assert_called_once_with("mydb", "SELECT * FROM users")

    async def test_execute_delegates_query_with_limit(self) -> None:
        """execute() should pass limit parameter to backend."""
        backend = MockDatabaseBackend()
        backend.query.return_value = {"rows": [], "row_count": 0}
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "mydb", "sql": "SELECT * FROM users", "limit": 10},
        )

        assert result.success is True
        backend.query.assert_called_once_with(
            "mydb",
            "SELECT * FROM users",
            limit=10,
        )

    async def test_execute_delegates_execute_statement_to_backend(self) -> None:
        """execute() should delegate execute_statement to the backend."""
        backend = MockDatabaseBackend()
        backend.execute_statement.return_value = {"rows_affected": 5}
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:execute_statement",
            {"database": "mydb", "sql": "DELETE FROM logs WHERE created_at < NOW()"},
        )

        assert result.success is True
        assert result.data == {"rows_affected": 5}
        backend.execute_statement.assert_called_once()

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """execute() should return error for unknown tool names."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:unknown_tool",
            {"database": "mydb"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_backend_error_propagates_as_tool_result_failure(self) -> None:
        """Backend exceptions should become ToolResult(success=False)."""
        backend = MockDatabaseBackend()
        backend.query.side_effect = Exception("Connection refused")
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "mydb", "sql": "SELECT 1"},
        )

        assert result.success is False
        assert result.error == "Connection refused"

    async def test_execute_statement_backend_error_propagates(self) -> None:
        """execute_statement backend errors should propagate as failures."""
        backend = MockDatabaseBackend()
        backend.execute_statement.side_effect = Exception("Permission denied")
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:execute_statement",
            {"database": "mydb", "sql": "DROP TABLE users"},
        )

        assert result.success is False
        assert result.error == "Permission denied"

    async def test_resolve_conditions_derives_database_name(self) -> None:
        """resolve_conditions should derive database:DatabaseName from params."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "production_db", "sql": "SELECT * FROM users"},
        )

        assert conditions.get("database:DatabaseName") == "production_db"

    async def test_resolve_conditions_extracts_table_from_select(self) -> None:
        """resolve_conditions should extract table name from SELECT."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "mydb", "sql": "SELECT * FROM users"},
        )

        assert conditions.get("database:TableName") == "users"

    async def test_resolve_conditions_extracts_table_from_select_with_where(
        self,
    ) -> None:
        """resolve_conditions should extract table from SELECT with WHERE."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {
                "database": "mydb",
                "sql": "SELECT id, name FROM orders WHERE status = 'pending'",
            },
        )

        assert conditions.get("database:TableName") == "orders"

    async def test_resolve_conditions_extracts_table_from_insert(self) -> None:
        """resolve_conditions should extract table name from INSERT."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:execute_statement",
            {
                "database": "mydb",
                "sql": "INSERT INTO orders (id, status) VALUES (1, 'new')",
            },
        )

        assert conditions.get("database:TableName") == "orders"

    async def test_resolve_conditions_extracts_table_from_update(self) -> None:
        """resolve_conditions should extract table name from UPDATE."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:execute_statement",
            {"database": "mydb", "sql": "UPDATE products SET price = 10 WHERE id = 1"},
        )

        assert conditions.get("database:TableName") == "products"

    async def test_resolve_conditions_extracts_table_from_delete(self) -> None:
        """resolve_conditions should extract table name from DELETE."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:execute_statement",
            {
                "database": "mydb",
                "sql": ("DELETE FROM logs WHERE created_at < '2024-01-01'"),
            },
        )

        assert conditions.get("database:TableName") == "logs"

    async def test_resolve_conditions_handles_table_with_underscores(self) -> None:
        """resolve_conditions should extract table names with underscores."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "mydb", "sql": "SELECT * FROM user_accounts"},
        )

        assert conditions.get("database:TableName") == "user_accounts"

    async def test_resolve_conditions_handles_table_with_numbers(self) -> None:
        """resolve_conditions should extract table names with numbers."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "mydb", "sql": "SELECT * FROM logs_2024"},
        )

        assert conditions.get("database:TableName") == "logs_2024"

    async def test_resolve_conditions_returns_empty_without_database(self) -> None:
        """resolve_conditions should handle missing database parameter."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {"sql": "SELECT 1"},
        )

        assert "database:DatabaseName" not in conditions

    async def test_resolve_conditions_returns_empty_without_sql(self) -> None:
        """resolve_conditions should handle missing sql parameter."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "mydb"},
        )

        assert "database:TableName" not in conditions

    async def test_resolve_conditions_handles_case_insensitive_sql(self) -> None:
        """resolve_conditions should handle SQL regardless of case."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        # Test lowercase SQL
        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "mydb", "sql": "select * from users"},
        )
        assert conditions.get("database:TableName") == "users"

        # Test mixed case SQL
        conditions = await module.resolve_conditions(
            "database:query",
            {"database": "mydb", "sql": "Select * From Users"},
        )
        assert conditions.get("database:TableName") == "users"

    async def test_execute_validates_missing_database(self) -> None:
        """execute() should return error for missing database parameter."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"sql": "SELECT 1"},
        )

        assert result.success is False
        assert "Missing required parameter: database" in result.error

    async def test_execute_validates_missing_sql(self) -> None:
        """execute() should return error for missing sql parameter."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "mydb"},
        )

        assert result.success is False
        assert "Missing required parameter: sql" in result.error

    async def test_execute_validates_empty_database(self) -> None:
        """execute() should return error for empty database name."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "", "sql": "SELECT 1"},
        )

        assert result.success is False
        assert "Database name must not be empty" in result.error

    async def test_execute_validates_empty_sql(self) -> None:
        """execute() should return error for empty SQL."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "mydb", "sql": ""},
        )

        assert result.success is False
        assert "SQL query must not be empty" in result.error

    async def test_execute_validates_whitespace_only_database(self) -> None:
        """execute() should return error for whitespace-only database name."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "   ", "sql": "SELECT 1"},
        )

        assert result.success is False
        assert "Database name must not be empty" in result.error

    async def test_execute_validates_whitespace_only_sql(self) -> None:
        """execute() should return error for whitespace-only SQL."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:query",
            {"database": "mydb", "sql": "   "},
        )

        assert result.success is False
        assert "SQL query must not be empty" in result.error

    async def test_execute_statement_validates_missing_database(self) -> None:
        """execute_statement should validate missing database parameter."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:execute_statement",
            {"sql": "DELETE FROM users"},
        )

        assert result.success is False
        assert "Missing required parameter: database" in result.error

    async def test_execute_statement_validates_missing_sql(self) -> None:
        """execute_statement should validate missing sql parameter."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        result = await module.execute(
            "database:execute_statement",
            {"database": "mydb"},
        )

        assert result.success is False
        assert "Missing required parameter: sql" in result.error

    def test_module_descriptor_docstrings_mention_security(self) -> None:
        """Module should document security considerations."""
        # Verify security documentation in module docstring
        from safe_agent.modules import database

        docstring = database.__doc__ or ""
        lower_doc = docstring.lower()
        assert "sql injection" in lower_doc or "parameterized" in lower_doc

    def test_backend_protocol_docstrings_mention_security(self) -> None:
        """Backend protocol should document security requirements."""
        from safe_agent.modules.database import DatabaseBackend

        docstring = DatabaseBackend.__doc__ or ""
        # Check protocol docstring or method docstrings
        lower_doc = docstring.lower()
        query_doc = DatabaseBackend.query.__doc__ or ""
        assert "parameterized" in lower_doc or "injection" in query_doc.lower()

    def test_execute_statement_description_mentions_privileged(self) -> None:
        """execute_statement tool description should mention privileged action."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        descriptor = module.describe()
        execute_tool = next(
            tool
            for tool in descriptor.tools
            if tool.name == "database:execute_statement"
        )

        assert "privileged" in execute_tool.description.lower()


class TestTableNameExtraction:
    """Tests for SQL table name extraction."""

    def test_extract_table_from_select_simple(self) -> None:
        """Should extract table from simple SELECT."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("SELECT * FROM users")
        assert table == "users"

    def test_extract_table_from_select_with_columns(self) -> None:
        """Should extract table from SELECT with column list."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("SELECT id, name, email FROM customers")
        assert table == "customers"

    def test_extract_table_from_select_with_where(self) -> None:
        """Should extract table from SELECT with WHERE clause."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name(
            "SELECT * FROM orders WHERE status = 'pending'"
        )
        assert table == "orders"

    def test_extract_table_from_insert(self) -> None:
        """Should extract table from INSERT statement."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name(
            "INSERT INTO products (id, name) VALUES (1, 'Widget')"
        )
        assert table == "products"

    def test_extract_table_from_update(self) -> None:
        """Should extract table from UPDATE statement."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name(
            "UPDATE inventory SET quantity = 10 WHERE id = 1"
        )
        assert table == "inventory"

    def test_extract_table_from_delete(self) -> None:
        """Should extract table from DELETE statement."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("DELETE FROM sessions WHERE expired = true")
        assert table == "sessions"

    def test_extract_table_handles_lowercase(self) -> None:
        """Should handle lowercase SQL."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("select * from my_table")
        assert table == "my_table"

    def test_extract_table_handles_uppercase(self) -> None:
        """Should handle uppercase SQL."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("SELECT * FROM MY_TABLE")
        assert table == "my_table"

    def test_extract_table_handles_mixed_case(self) -> None:
        """Should handle mixed case SQL."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("Select * From MyTable")
        assert table == "mytable"

    def test_extract_table_handles_extra_whitespace(self) -> None:
        """Should handle SQL with extra whitespace."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("SELECT   *   FROM   users")
        assert table == "users"

    def test_extract_table_handles_newlines(self) -> None:
        """Should handle SQL with newlines."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("""
            SELECT *
            FROM users
            WHERE active = true
        """)
        assert table == "users"

    def test_extract_table_returns_none_for_invalid_sql(self) -> None:
        """Should return None for SQL without recognizable table."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("SELECT 1")
        assert table is None

    def test_extract_table_returns_none_for_empty_string(self) -> None:
        """Should return None for empty SQL string."""
        backend = MockDatabaseBackend()
        module = DatabaseModule(backend)

        table = module._extract_table_name("")
        assert table is None
