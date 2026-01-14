# Copyright 2024 The Bazel Authors. All rights reserved.
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
# LINT.IfChange(forked_exports)
"""
Utility functions for C++ rules that don't depend on cc_common.

Only use those within C++ implementation. The others need to go through cc_common.
"""

CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES = [
    ("", "devtools/rust/cc_interop"),
    ("", "third_party/crubit"),
    ("", "tools/build_defs/clif"),
    ("", "third_party/bazel_rules/rules_cc"),
]

PRIVATE_STARLARKIFICATION_ALLOWLIST = [
    ("_builtins", ""),
    # Android rules
    ("", "tools/build_defs/android"),
    ("", "third_party/bazel_rules/rules_android"),
    ("build_bazel_rules_android", ""),
    ("rules_android", ""),
    # Apple rules
    ("", "third_party/bazel_rules/rules_apple"),
    ("apple_support", ""),
    ("rules_apple", ""),
    # C++ rules
    ("", "bazel_internal/test_rules/cc"),
    ("", "third_party/bazel_rules/rules_cc"),
    ("", "tools/build_defs/cc"),
    ("rules_cc", ""),
    # CUDA rules
    ("", "third_party/gpus/cuda"),
    # Go rules
    ("", "tools/build_defs/go"),
    # Java rules
    ("", "third_party/bazel_rules/rules_java"),
    ("rules_java", ""),
    # Objc rules
    ("", "tools/build_defs/objc"),
    # Protobuf rules
    ("", "third_party/protobuf"),
    ("protobuf", ""),
    ("com_google_protobuf", ""),
    # Rust rules
    ("", "rust/private"),
    ("rules_rust", "rust/private"),
    # Python rules
    ("", "third_party/bazel_rules/rules_python"),
    # Various
    ("", "research/colab"),
] + CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES

# LINT.ThenChange(@rules_cc//cc/common/cc_helper_internal.bzl:forked_exports)
