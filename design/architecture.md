# SafeAgentFramework — Design

A pluggable agent framework where **you** control exactly what the AI can do.

## Core Idea

The LLM is treated as untrusted input — like an HTTP request hitting a server.
It can attempt any tool call with any parameters. Your code decides whether to
execute it, based on policies you write. No prompt-based guardrails. No
self-governance. Code enforcement.

## Three Concepts

### 1. Modules

Modules are pluggable capabilities (filesystem, shell, email, etc.). Each module
is a Python class that:

- **Describes** its tools — name, parameters, and what IAM action/resource each
  tool maps to.
- **Resolves conditions** — derives context from tool parameters (e.g. file
  extension from a path) so policies can use them.
- **Executes** — performs the actual operation, only called after authorization.

```python
class BaseModule(ABC):

    @abstractmethod
    def describe(self) -> ModuleDescriptor:
        """Declare tools and their IAM metadata."""
        ...

    @abstractmethod
    async def resolve_conditions(self, tool_name: str, params: dict) -> dict[str, Any]:
        """Derive condition values from tool parameters."""
        ...

    @abstractmethod
    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Run the tool. Only called after authorization passes."""
        ...
```

A tool descriptor pairs the LLM-facing schema with the IAM-facing metadata.
All data models use Pydantic `BaseModel` for validation and serialization:

```python
class ToolDescriptor(BaseModel):
    name: str                        # Tool name exposed to the LLM
    description: str                 # Human-readable description
    parameters: dict                 # JSON Schema for parameters
    action: str                      # IAM action (e.g. "filesystem:ReadFile")
    resource_param: str | list[str]  # Which param(s) contain the IAM resource
    condition_keys: list[str]        # Condition keys this tool supports

class ModuleDescriptor(BaseModel):
    namespace: str                   # IAM namespace (e.g. "filesystem")
    description: str                 # Human-readable module description
    tools: list[ToolDescriptor]      # Tools provided by this module

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Modules are discovered via Python entry points — install a package, restart,
and the new tools are available (subject to policies). No core runtime changes.

```toml
[project.entry-points."safe_agent.modules"]
filesystem = "my_package.filesystem:FilesystemModule"
```

### 2. Policies

AWS IAM-inspired, deny-by-default. A policy is a list of statements that
allow or deny specific actions on specific resources, optionally with
conditions.

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowReadProjectFiles",
      "Effect": "Allow",
      "Action": ["filesystem:ReadFile", "filesystem:ListDirectory"],
      "Resource": ["/projects/*"]
    },
    {
      "Sid": "BlockSecrets",
      "Effect": "Deny",
      "Action": ["filesystem:*"],
      "Resource": ["*.env", "*.pem", "*.key"]
    }
  ]
}
```

**Evaluation rules** (same as AWS IAM):

1. Default: **deny**.
2. Collect statements whose Action and Resource match the request.
3. If any matching statement has `Effect: Deny` → **denied**. Explicit deny always wins.
4. If any matching statement has `Effect: Allow` (and conditions pass) → **allowed**.
5. Otherwise → **denied** (implicit deny, nothing matched).

The system fails closed. Missing or misconfigured policies result in denial.

**Conditions** add contextual constraints. Multiple conditions are ANDed;
multiple values within a condition are ORed.

```json
{
  "Condition": {
    "StringNotEquals": {
      "filesystem:FileExtension": [".env", ".pem"]
    }
  }
}
```

Supported operators: `StringEquals`, `StringNotEquals`, `StringLike`,
`StringNotLike`, `NumericEquals`, `NumericNotEquals`, `NumericLessThan`,
`NumericGreaterThan`, `NumericLessThanEquals`, `NumericGreaterThanEquals`.

Policies are loaded once at startup and frozen. No hot-reload, no runtime
modification. Changes require a restart.

### 3. The Code Gate

Every tool call passes through policy evaluation. There is exactly one path
from an LLM tool call to a module's `execute()`, and it goes through the
evaluator. No bypass, no override, no skip flag.

```
LLM requests tool call (untrusted)
        │
        ▼
Look up ToolDescriptor
        │
        ▼
Extract action + resource from descriptor
        │
        ▼
Module resolves conditions from params
        │
        ▼
Policy evaluator: allow or deny?
        │
   ┌────┴────┐
 DENY     ALLOW
   │         │
   ▼         ▼
Generic    module.execute()
error         │
              ▼
         Return result to LLM
```

What the LLM **never** sees: policy rules, denial reasons, which statement
matched, or whether denial was explicit or implicit. It gets tool definitions
(so it can construct calls) and a generic "Action denied" on failure. Policies
are invisible infrastructure.

## Design Principles

1. **Zero direct access.** The agent never touches system resources directly.
   Everything goes through a module.
2. **Deny by default.** No policy = no access.
3. **Code enforcement.** Authorization is a code gate, not a prompt instruction.
   The LLM and the enforcer are separate.
4. **Self-describing modules.** The runtime has no hardcoded knowledge of any
   module. Modules declare everything needed for tool registration and policy
   evaluation.
5. **Pluggable.** Add capabilities by installing a package. No runtime changes.

## Runtime Flow

```
User message → Gateway → Event Loop (per-session)
                              │
                              ├─ Assemble context
                              ├─ Call LLM
                              ├─ For each tool call:
                              │    ├─ Look up descriptor
                              │    ├─ Resolve conditions
                              │    ├─ Evaluate policy (CODE GATE)
                              │    ├─ If denied → generic error
                              │    └─ If allowed → execute, collect result
                              ├─ Append results to context
                              └─ Repeat until LLM returns text (or turn limit)
```

Sessions are isolated. Each session holds its own context and message history.
No cross-session state sharing.

## Security Model

- **Policies are not accessible to the agent.** The policy directory is outside
  every module's reachable namespace. There are no tools to list, read, or
  modify policies.
- **Policies are immutable after startup.** Loaded once, frozen, read-only.
- **Modules are discovered at startup only.** Mid-session package installs have
  no effect until restart.
- **Implementation-level sandboxing.** Modules enforce their own safety (e.g.
  filesystem path resolution prevents traversal) independently of policies.
  Sandboxing and authorization are separate layers.
- **Audit logging.** Every authorization decision is recorded (session, tool
  call, conditions, decision) to an append-only log outside the agent's reach.

## Adding a Module

A module is a self-contained capability. Adding one never requires changing
the core runtime. Here's the full process:

### 1. Implement the module class

Extend `BaseModule`. Set a namespace, describe your tools, resolve any
conditions policies might need, and implement the execution logic.

```python
class DatabaseModule(BaseModule):

    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            namespace="database",
            description="Query and write to databases",
            tools=[
                ToolDescriptor(
                    name="query",
                    description="Run a read-only SQL query",
                    parameters={
                        "type": "object",
                        "properties": {
                            "database": {"type": "string"},
                            "sql": {"type": "string"},
                        },
                        "required": ["database", "sql"],
                    },
                    action="database:Query",
                    resource_param="database",
                    condition_keys=["database:DatabaseName"],
                ),
            ],
        )

    async def resolve_conditions(self, tool_name: str, params: dict) -> dict:
        return {"database:DatabaseName": params.get("database", "")}

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        if tool_name == "query":
            # run the query against your backend
            return ToolResult(success=True, data={"rows": []})
        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
```

What each method does:

- **`describe()`** — returns the module's namespace and tool definitions.
  The runtime uses this to register tools with the LLM and to know how to
  build authorization requests. The `resource_param` tells the runtime
  which parameter contains the IAM resource (here, `"database"` — so the
  value of the `database` parameter becomes the resource being authorized).
- **`resolve_conditions()`** — called before policy evaluation. Derives
  condition values from the tool parameters so policies can reference them.
  If a policy says `"database:DatabaseName": ["analytics"]`, this method
  is what provides the actual database name for comparison.
- **`execute()`** — the actual operation. Only called after the code gate
  allows it.

### 2. Write a policy

Without a policy, the module exists but the agent can't use it (deny by
default). Write a policy that grants exactly the access you want:

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowAnalyticsQueries",
      "Effect": "Allow",
      "Action": ["database:Query"],
      "Resource": ["analytics"]
    }
  ]
}
```

This allows `database:Query` only when the resource is `"analytics"`.
Any other database, or any action not listed, is implicitly denied.

### 3. Register via entry point

Add the module to your package's entry points so the runtime discovers it
at startup:

```toml
[project.entry-points."safe_agent.modules"]
database = "my_package.database:DatabaseModule"
```

Install the package, restart the runtime, and the agent has a new `query`
tool — gated by your policy. No core code was modified.

### Key constraints

- **Modules cannot access the policy system.** There is no import path from
  a module to the evaluator, policy store, or audit log. A module receives
  parameters and returns results — it has no way to check, modify, or
  influence its own authorization.
- **Modules own their sandboxing.** A filesystem module should resolve paths
  and prevent traversal. A shell module should enforce timeouts. This is
  independent of policies — sandboxing is a second layer of defense inside
  the module itself.
- **Namespace collisions are rejected.** If two modules declare the same
  namespace, the runtime refuses to start. An administrator resolves the
  conflict.

## What This Is Not

- **Not prompt-based enforcement.** We never tell the LLM "you can only read
  files in /projects." That's a suggestion, not a control.
- **Not pre-filtered tool lists.** The LLM sees all tools. Policies are
  resource-dependent — the same tool might be allowed for one path and denied
  for another. Filtering at the tool level can't express this.
- **Not MCP.** The module protocol borrows MCP's self-describing tools and JSON
  Schema parameters but adds authorization as a first-class concern. MCP has
  no policy evaluation — connected = trusted. We don't assume that.
