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

"""Deprecated module path — use safe_agent.modules.web.browse instead."""

import warnings

from safe_agent.modules.web.browse import (
    DEFAULT_MAX_RESPONSE_SIZE,
    DEFAULT_TIMEOUT,
    MAX_SUMMARY_TEXT_LENGTH,
    MAX_TIMEOUT,
    WebBrowseModule,
    _extract_content_type,
    _extract_domain,
    _html_to_text,
)

warnings.warn(
    "safe_agent.modules.web_browse is deprecated; "
    "use safe_agent.modules.web.browse instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_MAX_RESPONSE_SIZE",
    "DEFAULT_TIMEOUT",
    "MAX_SUMMARY_TEXT_LENGTH",
    "MAX_TIMEOUT",
    "WebBrowseModule",
    "_extract_content_type",
    "_extract_domain",
    "_html_to_text",
]
