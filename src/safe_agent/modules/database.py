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

"""Database module for SafeAgent.

This module provides database query and execution capabilities through a pluggable
backend protocol. The framework defines the interface; a plugin provides the
concrete implementation (PostgreSQL, MySQL, SQLite, etc.).

Security Considerations:
    - SQL Injection Prevention: Backend implementations MUST use parameterized
      queries or prepared statements. The module passes raw SQL strings, so the
      backend is responsible for proper escaping and parameter binding.

    - Privileged Actions: The execute_statement tool performs write operations
      (INSERT, UPDATE, DELETE) and should be treated as a privileged action.
      Policy conditions should restrict access appropriately.

    - Query Result Exposure: Query results are returned in ToolResult.data and
      may be visible to the LLM. Avoid querying sensitive data without proper
      access controls.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol for database backend implementations.

    Implementations provide concrete integrations with database systems
    such as PostgreSQL, MySQL, SQLite, MongoDB, etc.

    Security Note:
        Backend implementations MUST use parameterized queries or prepared
        statements to prevent SQL injection attacks. Raw SQL strings received
        from the module should never be executed directly without proper
        parameter binding.

    Methods:
        query: Execute a read-only SQL query.
        execute_statement: Execute a write SQL statement (INSERT, UPDATE, DELETE).
    """

    async def query(self, database: str, sql: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a read-only SQL query.

        Args:
            database: The target database name or identifier.
            sql: The SQL query string (SELECT statements only).
            **kwargs: Backend-specific options (e.g., limit, timeout).

        Returns:
            Dict containing query results with at minimum:
                - "rows": List of row dicts (column -> value).
                - "row_count": Number of rows returned.

        Raises:
            Exception: If the query fails or attempts a write operation.
        """
        ...

    async def execute_statement(
        self,
        database: str,
        sql: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a write SQL statement (INSERT, UPDATE, DELETE).

        Args:
            database: The target database name or identifier.
            sql: The SQL statement string (INSERT, UPDATE, or DELETE).
            **kwargs: Backend-specific options (e.g., timeout).

        Returns:
            Dict containing execution results with at minimum:
                - "rows_affected": Number of rows modified.

        Raises:
            Exception: If the statement fails.
        """
        ...


class DatabaseModule(BaseModule):
    """Database module using the pluggable adapter pattern.

    This module exposes database tools to the SafeAgent framework while
    delegating actual database operations to an injected backend provider.

    Security Note:
        - SQL injection prevention is the backend's responsibility.
        - execute_statement is a privileged action - use policies to restrict.

    Attributes:
        _backend: The database backend implementation (private).
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        """Initialize the database module with a backend provider.

        Args:
            backend: An implementation of DatabaseBackend that handles
                actual database operations for a specific platform.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the database module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="database",
            description="Execute SQL queries and statements on databases.",
            tools=[
                ToolDescriptor(
                    name="database:query",
                    description="Run a read-only SQL query on a database.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "description": "The target database name.",
                            },
                            "sql": {
                                "type": "string",
                                "description": "The SQL query string (SELECT only).",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "Optional limit on number of rows to return."
                                ),
                            },
                        },
                        "required": ["database", "sql"],
                        "additionalProperties": False,
                    },
                    action="database:Query",
                    resource_param=["database"],
                    condition_keys=[
                        "database:DatabaseName",
                        "database:TableName",
                    ],
                ),
                ToolDescriptor(
                    name="database:execute_statement",
                    description=(
                        "Run a write statement (INSERT, UPDATE, DELETE) on a database. "
                        "This is a privileged action."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "description": "The target database name.",
                            },
                            "sql": {
                                "type": "string",
                                "description": (
                                    "The SQL statement (INSERT, UPDATE, or DELETE)."
                                ),
                            },
                        },
                        "required": ["database", "sql"],
                        "additionalProperties": False,
                    },
                    action="database:ExecuteStatement",
                    resource_param=["database"],
                    condition_keys=[
                        "database:DatabaseName",
                        "database:TableName",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve database condition values from parameters.

        Derives condition keys for policy evaluation:
            - database:DatabaseName: The target database name.
            - database:TableName: Best-effort extraction from SQL query.

        Table name extraction is best-effort and supports common patterns:
            - SELECT ... FROM table_name
            - INSERT INTO table_name
            - UPDATE table_name
            - DELETE FROM table_name

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        conditions: dict[str, Any] = {}

        database = params.get("database")
        if database is not None:
            conditions["database:DatabaseName"] = str(database)

        sql = params.get("sql")
        if sql is not None:
            table_name = self._extract_table_name(str(sql))
            if table_name is not None:
                conditions["database:TableName"] = table_name

        return conditions

    def _extract_table_name(self, sql: str) -> str | None:
        """Extract table name from SQL query using best-effort parsing.

        Supports common SQL patterns:
            - SELECT ... FROM table_name [WHERE ...]
            - INSERT INTO table_name [(columns)] VALUES ...
            - UPDATE table_name SET ...
            - DELETE FROM table_name [WHERE ...]

        Args:
            sql: The SQL query string.

        Returns:
            The extracted table name, or None if extraction fails.
        """
        # Normalize SQL: collapse whitespace, convert to uppercase for matching
        normalized = " ".join(sql.upper().split())

        # Pattern 1: SELECT ... FROM table_name
        # Matches: SELECT * FROM users, SELECT id FROM users WHERE ...
        select_match = re.search(
            r"\bFROM\s+([A-Z_][A-Z0-9_]*)",
            normalized,
        )
        if select_match:
            return select_match.group(1).lower()

        # Pattern 2: INSERT INTO table_name
        # Matches: INSERT INTO users (id) VALUES (1)
        insert_match = re.search(
            r"\bINSERT\s+INTO\s+([A-Z_][A-Z0-9_]*)",
            normalized,
        )
        if insert_match:
            return insert_match.group(1).lower()

        # Pattern 3: UPDATE table_name SET
        # Matches: UPDATE users SET name = 'x' WHERE ...
        update_match = re.search(
            r"\bUPDATE\s+([A-Z_][A-Z0-9_]*)\s+SET\b",
            normalized,
        )
        if update_match:
            return update_match.group(1).lower()

        # Pattern 4: DELETE FROM table_name
        # Matches: DELETE FROM users WHERE ...
        delete_match = re.search(
            r"\bDELETE\s+FROM\s+([A-Z_][A-Z0-9_]*)",
            normalized,
        )
        if delete_match:
            return delete_match.group(1).lower()

        return None

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a database tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("database:")
            if normalized_tool == "query":
                return await self._query(params)
            if normalized_tool == "execute_statement":
                return await self._execute_statement(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _query(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate query execution to the backend.

        Security Note:
            Logs the database and SQL at DEBUG level. Query results are NOT logged.
        """
        database = params.get("database")
        if database is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: database",
            )

        sql = params.get("sql")
        if sql is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: sql",
            )

        database = str(database)
        sql = str(sql)
        if not database or not database.strip():
            return ToolResult(
                success=False,
                error="Database name must not be empty",
            )
        if not sql or not sql.strip():
            return ToolResult(
                success=False,
                error="SQL query must not be empty",
            )

        logger.debug("Database query: database=%s sql=%s", database, sql)

        kwargs: dict[str, Any] = {}
        if "limit" in params:
            kwargs["limit"] = params["limit"]

        result = await self._backend.query(database, sql, **kwargs)
        return ToolResult(success=True, data=result)

    async def _execute_statement(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate statement execution to the backend.

        Security Note:
            This is a privileged action that performs write operations.
            Logs the database and SQL at INFO level for audit trail.
            Result data is NOT logged.
        """
        database = params.get("database")
        if database is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: database",
            )

        sql = params.get("sql")
        if sql is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: sql",
            )

        database = str(database)
        sql = str(sql)
        if not database or not database.strip():
            return ToolResult(
                success=False,
                error="Database name must not be empty",
            )
        if not sql or not sql.strip():
            return ToolResult(
                success=False,
                error="SQL statement must not be empty",
            )

        # Log write operations at INFO level for audit trail
        logger.info("Database execute_statement: database=%s sql=%s", database, sql)

        result = await self._backend.execute_statement(database, sql)
        return ToolResult(success=True, data=result)
