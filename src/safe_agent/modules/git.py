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

"""Git version control module for SafeAgent.

Provides controlled git operations with configurable working directory,
timeout limits, and output size limits.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Default timeout for git operations (seconds)
DEFAULT_TIMEOUT = 60.0

# Maximum timeout allowed (5 minutes)
MAX_TIMEOUT = 300.0

# Maximum output size (1MB)
MAX_OUTPUT_SIZE = 1024 * 1024


class GitModule(BaseModule):
    """Provide controlled git operations with timeout and output limits.

    This module wraps the git CLI and provides a safe, structured API for
    common git operations including clone, pull, push, commit, branch,
    merge, status, log, diff, and tag.

    All operations return structured results with typed data or clear
    error messages. Git operations are executed via subprocess with
    configurable timeouts and output limits.
    """

    def __init__(
        self,
        working_directory: Path | None = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        max_timeout: float = MAX_TIMEOUT,
        max_output_size: int = MAX_OUTPUT_SIZE,
    ) -> None:
        """Initialize the Git module.

        Args:
            working_directory: Optional working directory for git operations.
                If None, uses the current working directory.
            default_timeout: Default timeout for git operations in seconds.
            max_timeout: Maximum allowed timeout for any operation.
            max_output_size: Maximum output size in bytes.

        Raises:
            ValueError: If timeout values are invalid.
        """
        self.working_directory = (
            working_directory.resolve()
            if working_directory is not None
            else Path.cwd().resolve()
        )
        if max_timeout <= 0:
            raise ValueError("max_timeout must be > 0")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be > 0")
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.max_output_size = max_output_size

    def describe(self) -> ModuleDescriptor:
        """Return the git module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="git",
            description=(
                "Git version control operations with timeout and output limits."
            ),
            tools=[
                ToolDescriptor(
                    name="git:clone",
                    description="Clone a git repository.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "destination": {"type": "string"},
                            "branch": {"type": "string"},
                            "depth": {"type": "integer"},
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    action="git:Clone",
                    resource_param=["url"],
                    condition_keys=["git:RepositoryUrl"],
                ),
                ToolDescriptor(
                    name="git:pull",
                    description="Pull changes from a remote repository.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "remote": {"type": "string", "default": "origin"},
                            "branch": {"type": "string"},
                            "rebase": {"type": "boolean", "default": False},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="git:Pull",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:push",
                    description="Push changes to a remote repository.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "remote": {"type": "string", "default": "origin"},
                            "branch": {"type": "string"},
                            "force": {"type": "boolean", "default": False},
                            "set_upstream": {"type": "boolean", "default": False},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="git:Push",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:commit",
                    description="Commit staged changes.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "allow_empty": {"type": "boolean", "default": False},
                            "amend": {"type": "boolean", "default": False},
                        },
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                    action="git:Commit",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:branch",
                    description="Create, list, or delete branches.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list", "create", "delete", "current"],
                            },
                            "name": {"type": "string"},
                            "force": {"type": "boolean", "default": False},
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    action="git:Branch",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:merge",
                    description="Merge a branch into the current branch.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "branch": {"type": "string"},
                            "no_ff": {"type": "boolean", "default": False},
                            "message": {"type": "string"},
                        },
                        "required": ["branch"],
                        "additionalProperties": False,
                    },
                    action="git:Merge",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:status",
                    description="Show the working tree status.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "porcelain": {"type": "boolean", "default": False},
                            "short": {"type": "boolean", "default": False},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="git:Status",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:log",
                    description="Show commit logs.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "max_count": {"type": "integer"},
                            "oneline": {"type": "boolean", "default": False},
                            "branch": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="git:Log",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:diff",
                    description=(
                        "Show changes between commits, commit and working tree, etc."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "commit1": {"type": "string"},
                            "commit2": {"type": "string"},
                            "path": {"type": "string"},
                            "staged": {"type": "boolean", "default": False},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="git:Diff",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
                ToolDescriptor(
                    name="git:tag",
                    description="Create, list, or delete tags.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list", "create", "delete"],
                            },
                            "name": {"type": "string"},
                            "message": {"type": "string"},
                            "commit": {"type": "string"},
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    action="git:Tag",
                    resource_param=[],
                    condition_keys=["git:WorkingDirectory"],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve runtime conditions for policy evaluation."""
        conditions: dict[str, Any] = {}

        # Add working directory condition
        conditions["git:WorkingDirectory"] = self._working_directory_string()

        # Add repository URL for clone operations
        if tool_name == "git:clone":
            url = params.get("url", "")
            conditions["git:RepositoryUrl"] = str(url)

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute a git tool invocation."""
        normalized_tool = tool_name.removeprefix("git:")

        try:
            if normalized_tool == "clone":
                return await self._clone(params)
            if normalized_tool == "pull":
                return await self._pull(params)
            if normalized_tool == "push":
                return await self._push(params)
            if normalized_tool == "commit":
                return await self._commit(params)
            if normalized_tool == "branch":
                return await self._branch(params)
            if normalized_tool == "merge":
                return await self._merge(params)
            if normalized_tool == "status":
                return await self._status(params)
            if normalized_tool == "log":
                return await self._log(params)
            if normalized_tool == "diff":
                return await self._diff(params)
            if normalized_tool == "tag":
                return await self._tag(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive catch
            return ToolResult(success=False, error=str(exc))

    # --- Internal implementation methods ---

    async def _clone(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Clone a repository."""
        url = params.get("url")
        if not url:
            return ToolResult(success=False, error="url is required")

        args = ["clone", str(url)]

        destination = params.get("destination")
        if destination:
            args.append(str(destination))

        branch = params.get("branch")
        if branch:
            args.extend(["--branch", str(branch)])

        depth = params.get("depth")
        if depth:
            args.extend(["--depth", str(depth)])

        result = await self._run_git(args, cwd=None)  # Clone doesn't need cwd

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git clone failed"),
            )

        return ToolResult(
            success=True,
            data={
                "url": str(url),
                "destination": str(destination) if destination else None,
                "output": result.get("stdout", ""),
            },
        )

    async def _pull(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Pull changes from remote."""
        args = ["pull"]

        remote = params.get("remote", "origin")
        args.append(str(remote))

        branch = params.get("branch")
        if branch:
            args.append(str(branch))

        rebase = params.get("rebase", False)
        if rebase:
            args.append("--rebase")

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git pull failed"),
            )

        return ToolResult(
            success=True,
            data={
                "remote": str(remote),
                "branch": str(branch) if branch else None,
                "output": result.get("stdout", ""),
            },
        )

    async def _push(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Push changes to remote."""
        args = ["push"]

        remote = params.get("remote", "origin")
        args.append(str(remote))

        branch = params.get("branch")
        if branch:
            args.append(str(branch))

        force = params.get("force", False)
        if force:
            args.append("--force")

        set_upstream = params.get("set_upstream", False)
        if set_upstream:
            args.append("--set-upstream")

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git push failed"),
            )

        return ToolResult(
            success=True,
            data={
                "remote": str(remote),
                "branch": str(branch) if branch else None,
                "output": result.get("stdout", ""),
            },
        )

    async def _commit(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Commit staged changes."""
        message = params.get("message")
        if not message:
            return ToolResult(success=False, error="message is required")

        args = ["commit", "-m", str(message)]

        allow_empty = params.get("allow_empty", False)
        if allow_empty:
            args.append("--allow-empty")

        amend = params.get("amend", False)
        if amend:
            args.append("--amend")

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git commit failed"),
            )

        return ToolResult(
            success=True,
            data={
                "message": str(message),
                "output": result.get("stdout", ""),
            },
        )

    async def _branch(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Branch operations: list, create, delete, current."""
        action = params.get("action")
        if not action:
            return ToolResult(success=False, error="action is required")

        if action == "list":
            return await self._branch_list()
        if action == "current":
            return await self._branch_current()
        if action == "create":
            return await self._branch_create(params)
        if action == "delete":
            return await self._branch_delete(params)

        return ToolResult(success=False, error=f"Unknown branch action: {action}")

    async def _branch_list(self) -> ToolResult[dict[str, Any]]:
        """List all branches."""
        args = ["branch", "-a"]
        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git branch list failed"),
            )

        branches = self._parse_branch_list(result.get("stdout", ""))
        return ToolResult(
            success=True,
            data={
                "branches": branches,
                "output": result.get("stdout", ""),
            },
        )

    async def _branch_current(self) -> ToolResult[dict[str, Any]]:
        """Get the current branch name."""
        args = ["branch", "--show-current"]
        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git branch current failed"),
            )

        branch_name = result.get("stdout", "").strip()
        return ToolResult(
            success=True,
            data={
                "branch": branch_name,
                "output": result.get("stdout", ""),
            },
        )

    async def _branch_create(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Create a new branch."""
        name = params.get("name")
        if not name:
            return ToolResult(success=False, error="name is required for create action")

        args = ["branch", str(name)]

        force = params.get("force", False)
        if force:
            args.append("--force")

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git branch create failed"),
            )

        return ToolResult(
            success=True,
            data={
                "branch": str(name),
                "action": "create",
                "output": result.get("stdout", ""),
            },
        )

    async def _branch_delete(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Delete a branch."""
        name = params.get("name")
        if not name:
            return ToolResult(success=False, error="name is required for delete action")

        args = ["branch", "-d", str(name)]

        force = params.get("force", False)
        if force:
            args = ["branch", "-D", str(name)]

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git branch delete failed"),
            )

        return ToolResult(
            success=True,
            data={
                "branch": str(name),
                "action": "delete",
                "output": result.get("stdout", ""),
            },
        )

    async def _merge(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Merge a branch."""
        branch = params.get("branch")
        if not branch:
            return ToolResult(success=False, error="branch is required")

        args = ["merge", str(branch)]

        no_ff = params.get("no_ff", False)
        if no_ff:
            args.append("--no-ff")

        message = params.get("message")
        if message:
            args.extend(["-m", str(message)])

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git merge failed"),
            )

        return ToolResult(
            success=True,
            data={
                "branch": str(branch),
                "output": result.get("stdout", ""),
            },
        )

    async def _status(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Get working tree status."""
        args = ["status"]

        porcelain = params.get("porcelain", False)
        if porcelain:
            args.append("--porcelain")

        short = params.get("short", False)
        if short:
            args.append("--short")

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git status failed"),
            )

        status_data = self._parse_status(result.get("stdout", ""), porcelain=porcelain)
        return ToolResult(
            success=True,
            data={
                "status": status_data,
                "output": result.get("stdout", ""),
            },
        )

    async def _log(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Show commit logs."""
        args = ["log"]

        max_count = params.get("max_count")
        if max_count:
            args.extend(["-n", str(max_count)])

        oneline = params.get("oneline", False)
        if oneline:
            args.append("--oneline")

        branch = params.get("branch")
        if branch:
            args.append(str(branch))

        path = params.get("path")
        if path:
            args.extend(["--", str(path)])

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git log failed"),
            )

        commits = self._parse_log(result.get("stdout", ""), oneline=oneline)
        return ToolResult(
            success=True,
            data={
                "commits": commits,
                "output": result.get("stdout", ""),
            },
        )

    async def _diff(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Show changes."""
        args = ["diff"]

        staged = params.get("staged", False)
        if staged:
            args.append("--staged")

        commit1 = params.get("commit1")
        if commit1:
            args.append(str(commit1))

        commit2 = params.get("commit2")
        if commit2:
            args.append(str(commit2))

        path = params.get("path")
        if path:
            args.extend(["--", str(path)])

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git diff failed"),
            )

        return ToolResult(
            success=True,
            data={
                "diff": result.get("stdout", ""),
                "output": result.get("stdout", ""),
            },
        )

    async def _tag(self, params: dict[str, Any]) -> ToolResult[dict[str, Any]]:
        """Tag operations: list, create, delete."""
        action = params.get("action")
        if not action:
            return ToolResult(success=False, error="action is required")

        if action == "list":
            return await self._tag_list()
        if action == "create":
            return await self._tag_create(params)
        if action == "delete":
            return await self._tag_delete(params)

        return ToolResult(success=False, error=f"Unknown tag action: {action}")

    async def _tag_list(self) -> ToolResult[dict[str, Any]]:
        """List all tags."""
        args = ["tag", "-l"]
        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git tag list failed"),
            )

        tags = [t.strip() for t in result.get("stdout", "").split("\n") if t.strip()]
        return ToolResult(
            success=True,
            data={
                "tags": tags,
                "output": result.get("stdout", ""),
            },
        )

    async def _tag_create(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Create a tag."""
        name = params.get("name")
        if not name:
            return ToolResult(success=False, error="name is required for create action")

        args = ["tag", str(name)]

        message = params.get("message")
        if message:
            args.extend(["-a", "-m", str(message)])

        commit = params.get("commit")
        if commit:
            args.append(str(commit))

        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git tag create failed"),
            )

        return ToolResult(
            success=True,
            data={
                "tag": str(name),
                "action": "create",
                "output": result.get("stdout", ""),
            },
        )

    async def _tag_delete(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Delete a tag."""
        name = params.get("name")
        if not name:
            return ToolResult(success=False, error="name is required for delete action")

        args = ["tag", "-d", str(name)]
        result = await self._run_git(args)

        if not result["success"]:
            return ToolResult(
                success=False,
                error=result.get("error", "git tag delete failed"),
            )

        return ToolResult(
            success=True,
            data={
                "tag": str(name),
                "action": "delete",
                "output": result.get("stdout", ""),
            },
        )

    # --- Helper methods ---

    async def _run_git(
        self,
        args: list[str],
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Run a git command and return structured result.

        Args:
            args: Git arguments (without 'git' prefix).
            cwd: Working directory for the command.

        Returns:
            Dictionary with success, stdout, stderr, return_code, and error.
        """
        full_args = ["git", *args]
        effective_cwd = cwd if cwd is not None else self._cwd()

        try:
            process = await asyncio.create_subprocess_exec(
                *full_args,
                cwd=effective_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return {"success": False, "error": "git executable not found"}
        except Exception as exc:  # pragma: no cover - defensive catch
            return {"success": False, "error": str(exc)}

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.default_timeout,
            )
        except TimeoutError:
            self._safe_kill(process)
            await process.communicate()
            return {"success": False, "error": "Command timed out"}
        except Exception as exc:  # pragma: no cover - defensive catch
            self._safe_kill(process)
            await process.communicate()
            return {"success": False, "error": str(exc)}

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        # Truncate output if needed
        stdout_truncated = False
        stderr_truncated = False

        if len(stdout_text) > self.max_output_size:
            stdout_text = stdout_text[: self.max_output_size]
            stdout_truncated = True

        if len(stderr_text) > self.max_output_size:
            stderr_text = stderr_text[: self.max_output_size]
            stderr_truncated = True

        success = process.returncode == 0
        result: dict[str, Any] = {
            "success": success,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "return_code": process.returncode,
        }

        if stdout_truncated or stderr_truncated:
            result["output_truncated"] = True

        # Include stderr in error for non-zero return codes
        if not success:
            error_msg = (
                stderr_text.strip() or f"git exited with code {process.returncode}"
            )
            result["error"] = error_msg

        return result

    def _cwd(self) -> str | None:
        """Return the configured working directory."""
        if self.working_directory is None:
            return None
        return str(self.working_directory)

    def _working_directory_string(self) -> str:
        """Return the effective working directory as a string."""
        if self.working_directory is not None:
            return str(self.working_directory)
        return str(Path.cwd())

    def _safe_kill(self, process: asyncio.subprocess.Process) -> None:
        """Kill process safely, ignoring ProcessLookupError."""
        try:
            process.kill()
        except ProcessLookupError:
            pass  # Already terminated
        except OSError:  # pragma: no cover - defensive catch
            pass

    def _parse_branch_list(self, output: str) -> list[dict[str, str | bool]]:
        """Parse git branch -a output into structured data."""
        branches: list[dict[str, str | bool]] = []
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            current = line.startswith("* ")
            name = line.lstrip("* ").strip()

            # Handle remote branches
            is_remote = name.startswith("remotes/")
            branches.append(
                {
                    "name": name,
                    "current": current,
                    "is_remote": is_remote,
                }
            )

        return branches

    def _parse_status(
        self,
        output: str,
        porcelain: bool,
    ) -> dict[str, Any]:
        """Parse git status output into structured data."""
        if porcelain:
            # Porcelain format: XY PATH (XY are index and worktree status)
            files: list[dict[str, str]] = []
            for line in output.split("\n"):
                if not line:
                    continue
                # XY are two characters, followed by space(s) and path
                xy = line[:2]
                # Skip the space separator (can be 1 or more spaces)
                path = line[2:].lstrip()
                files.append(
                    {
                        "status": xy,
                        "path": path,
                    }
                )
            return {"files": files}

        # Regular status format - return raw output
        return {"raw": output}

    def _parse_log(
        self,
        output: str,
        oneline: bool,
    ) -> list[dict[str, str]]:
        """Parse git log output into structured data."""
        if oneline:
            # Oneline format: HASH MESSAGE
            commits: list[dict[str, str]] = []
            for line in output.split("\n"):
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    commits.append(
                        {
                            "hash": parts[0],
                            "message": parts[1],
                        }
                    )
                elif parts:
                    commits.append(
                        {
                            "hash": parts[0],
                            "message": "",
                        }
                    )
            return commits

        # Full format - return raw output for now
        return [{"raw": output}]
