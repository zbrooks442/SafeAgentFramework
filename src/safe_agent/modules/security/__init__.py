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

"""Security modules sub-package — vault and remote SSH modules."""

from safe_agent.modules.security.remote_ssh import RemoteSSHModule, SSHCredential
from safe_agent.modules.security.vault import VaultBackend, VaultModule

__all__ = [
    "RemoteSSHModule",
    "SSHCredential",
    "VaultBackend",
    "VaultModule",
]
