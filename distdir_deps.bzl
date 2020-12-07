# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""List the distribution dependencies we need to build Bazel."""

DIST_DEPS = {
    # This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    "rules_cc": {
        "archive": "b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        "sha256": "d0c573b94a6ef20ef6ff20154a23d0efcb409fb0e1ff0979cec318dfe42f0cdd",
        "strip_prefix": "rules_cc-b1c40e1de81913a3c40e5948f78719c28152486d",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
            "https://github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        ],
        "need_in_test_WORKSPACE": True,
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.2.4.tar.gz",
        "sha256": "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
        ],
    },
}
