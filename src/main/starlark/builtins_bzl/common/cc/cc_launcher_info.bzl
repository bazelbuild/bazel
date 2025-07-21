# Copyright 2025 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provider that signals that rules that use launchers can use this target as the launcher."""

load(":common/cc/cc_helper_internal.bzl", "wrap_with_check_private_api")

def _cc_launcher_info_constructor(cc_info, compilation_outputs):
    return dict(
        cc_info = wrap_with_check_private_api(cc_info),
        compilation_outputs = wrap_with_check_private_api(compilation_outputs),
    )

CcLauncherInfo, _ = provider(
    doc = "Provider that signals that rules that use launchers can use this target as the launcher.",
    fields = {
        "cc_info": "The CcInfo provider of the launcher.",
        "compilation_outputs": "The CcCompilationOutputs of the launcher.",
    },
    init = _cc_launcher_info_constructor,
)
