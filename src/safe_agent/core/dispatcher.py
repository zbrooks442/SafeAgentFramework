"""ToolDispatcher: the single enforcement point for all tool calls."""

from __future__ import annotations

import logging
from typing import Any

from safe_agent.core.audit import AuditEntry, AuditLogger
from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.models import AuthorizationRequest, Decision
from safe_agent.modules.base import ToolResult
from safe_agent.modules.registry import ModuleRegistry

logger = logging.getLogger(__name__)

# Single opaque error string returned on any dispatch failure.
# Identical for "unknown tool" and "denied" to prevent callers from using the
# error message to probe which tool names exist in the registry.
_DISPATCH_FAILED = "Dispatch failed"


class ToolDispatcher:
    """The single enforcement point where every LLM tool call is authorised.

    There is exactly one code path from an LLM tool call to a module's
    ``execute()`` method, and it passes through policy evaluation for every
    affected resource. There is no bypass, no override flag, and no skip path.

    Evaluation follows these steps for every ``dispatch()`` call:

    1. Look up the tool in the registry. Unknown tools are rejected *and*
       audited immediately — registry probing always leaves a trace.
    2. Extract resource(s) from ``params`` using ``descriptor.resource_param``.
       **Contract:** if ``resource_param`` is empty, evaluation proceeds against
       a single empty-string resource (``""``). The policy must explicitly allow
       ``resource=""`` for such tools — there is no implicit pass.
    3. Call ``module.resolve_conditions()`` to obtain runtime context values.
    4. For each resource: build an :class:`~safe_agent.iam.models.AuthorizationRequest`
       and call :meth:`~safe_agent.iam.evaluator.PolicyEvaluator.evaluate`.
    5. Log every decision via :class:`~safe_agent.core.audit.AuditLogger`.
       Every outcome — allowed, denied, unknown-tool, internal error — produces
       an audit entry. There is no silent dispatch path.
    6. On any internal exception (``resolve_conditions``, ``evaluate``,
       ``execute``, or ``log``): log a ``DENIED_IMPLICIT`` entry and return a
       generic failure. Exceptions never propagate to the caller.
    7. If audit logging fails at any point, dispatch **fails closed** — the
       tool is not executed and ``"Dispatch failed"`` is returned. An audit gap
       is a security event; execution without a record is not acceptable.
    8. If **any** resource is denied → return the opaque ``"Dispatch failed"``
       error. No policy detail, resource name, or statement info is exposed.
    9. If **all** resources are allowed → call ``module.execute()`` and return
       the result. A ``Decision.ALLOWED`` audit entry with ``"__executed__"``
       is written after successful execution.

    Args:
        registry: The :class:`~safe_agent.modules.registry.ModuleRegistry`
            containing all registered modules.
        evaluator: The :class:`~safe_agent.iam.evaluator.PolicyEvaluator`
            for authorisation decisions.
        audit_logger: The :class:`~safe_agent.core.audit.AuditLogger` for
            recording every decision.

    Example::

        dispatcher = ToolDispatcher(registry, evaluator, audit_logger)
        result = await dispatcher.dispatch(
            "fs:ReadFile", {"path": "/etc/hosts"}, "session-1"
        )
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        evaluator: PolicyEvaluator,
        audit_logger: AuditLogger,
    ) -> None:
        """Initialise the dispatcher.

        Args:
            registry: Module registry for tool lookup.
            evaluator: Policy evaluator for authorisation decisions.
            audit_logger: Audit logger for recording every decision.
        """
        self._registry = registry
        self._evaluator = evaluator
        self._audit_logger = audit_logger

    def _log_or_fail(self, entry: AuditEntry) -> bool:
        """Append *entry* to the audit log.

        Returns ``True`` on success, ``False`` if logging failed.

        Unlike a "log-safe-and-swallow" helper, callers are expected to treat
        a ``False`` return as a hard failure: dispatch must not proceed without
        an audit record. Callers should return :data:`_DISPATCH_FAILED` when
        this method returns ``False``.

        Args:
            entry: The :class:`AuditEntry` to persist.

        Returns:
            ``True`` if the entry was written successfully; ``False`` otherwise.
        """
        try:
            self._audit_logger.log(entry)
            return True
        except Exception:
            logger.exception(
                "safe_agent.dispatcher: audit logging failed for tool '%s' "
                "session '%s' — THIS IS A SECURITY EVENT; failing closed",
                entry.tool_name,
                entry.session_id,
            )
            return False

    async def dispatch(
        self,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
    ) -> ToolResult:
        """Authorise and execute a tool call.

        Every outcome — allowed, denied, unknown tool, or internal error —
        produces an audit log entry. No dispatch path is silent. If audit
        logging itself fails, dispatch **fails closed** — the tool is not
        executed.

        Args:
            tool_name: Fully-qualified tool name (e.g. ``"fs:ReadFile"``).
            params: Raw input parameters for the tool.
            session_id: Non-empty identifier for the calling session, used in
                audit logs.

        Returns:
            A :class:`~safe_agent.modules.base.ToolResult`. On any failure
            the error field is the opaque string ``"Dispatch failed"`` — no
            policy, resource, or internal detail is exposed.

        Raises:
            ValueError: If *session_id* is empty.
        """
        if not session_id:
            raise ValueError("session_id must be a non-empty string")

        timestamp = AuditLogger.now_iso()

        # Step 1 — look up the tool; audit and reject if unknown.
        lookup = self._registry.get_tool(tool_name)
        if lookup is None:
            self._log_or_fail(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions={},
                    decision=Decision.DENIED_IMPLICIT,
                    matched_statements=["__unknown_tool__"],
                )
            )
            # Return the same error regardless of whether audit succeeded —
            # an unknown tool is always rejected.
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        module, descriptor = lookup

        # Step 2 — extract resources from params.
        # Contract: if resource_param is empty, evaluate against "" — the
        # policy must explicitly allow resource="" for no-resource tools.
        resources: list[str] = []
        for param_name in descriptor.resource_param:
            value = params.get(param_name)
            if value is not None:
                resources.append(str(value))
        if not resources:
            resources = [""]

        # Step 3 — resolve runtime conditions.
        resolved_conditions: dict[str, Any] = {}
        try:
            resolved_conditions = await module.resolve_conditions(tool_name, params)
        except Exception:
            logger.exception(
                "safe_agent.dispatcher: resolve_conditions raised for tool '%s'",
                tool_name,
            )
            self._log_or_fail(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions={},
                    decision=Decision.DENIED_IMPLICIT,
                    matched_statements=["__internal_error__"],
                )
            )
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        # Steps 4 & 5 — evaluate each resource and log every decision.
        # Fail closed: if audit logging fails for any entry, abort immediately.
        denied = False
        for resource in resources:
            try:
                request = AuthorizationRequest(
                    action=descriptor.action,
                    resource=resource,
                    context=resolved_conditions,
                )
                eval_result = self._evaluator.evaluate(request)
            except Exception:
                logger.exception(
                    "safe_agent.dispatcher: evaluator raised for tool '%s' "
                    "resource '%s'",
                    tool_name,
                    resource,
                )
                self._log_or_fail(
                    AuditEntry(
                        session_id=session_id,
                        timestamp=timestamp,
                        tool_name=tool_name,
                        params=params,
                        resolved_conditions=resolved_conditions,
                        decision=Decision.DENIED_IMPLICIT,
                        matched_statements=["__internal_error__"],
                    )
                )
                # Stop on first evaluator error — avoids flooding the audit log
                # with repeated __internal_error__ entries for the same fault.
                denied = True
                break

            logged = self._log_or_fail(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions=resolved_conditions,
                    decision=eval_result.decision,
                    matched_statements=[
                        stmt.sid for stmt in eval_result.matched_statements
                    ],
                )
            )
            if not logged:
                # Audit failure: fail closed — do not execute without a record.
                return ToolResult(success=False, error=_DISPATCH_FAILED)

            if eval_result.decision != Decision.ALLOWED:
                # Intentional: no break here. We continue evaluating remaining
                # resources so every decision is audited, giving a complete
                # picture of what was denied and why.
                denied = True

        # Step 7/8 — deny if any resource was denied, with no policy details.
        if denied:
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        # Step 9 — all resources allowed; execute.
        try:
            result = await module.execute(tool_name, params)
        except Exception:
            logger.exception(
                "safe_agent.dispatcher: module.execute raised for tool '%s'",
                tool_name,
            )
            self._log_or_fail(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions=resolved_conditions,
                    decision=Decision.DENIED_IMPLICIT,
                    matched_statements=["__execute_error__"],
                )
            )
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        # Write an execution-occurred audit entry so the log shows not just
        # policy decisions but confirmed tool executions.
        self._log_or_fail(
            AuditEntry(
                session_id=session_id,
                timestamp=AuditLogger.now_iso(),
                tool_name=tool_name,
                params=params,
                resolved_conditions=resolved_conditions,
                decision=Decision.ALLOWED,
                matched_statements=["__executed__"],
            )
        )
        return result
