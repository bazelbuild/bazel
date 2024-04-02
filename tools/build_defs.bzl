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

"""Utility for compiling at Java 8."""

_java_language_version_8_transition = transition(
    implementation = lambda settings, attr: {
        "//command_line_option:java_language_version": "8",
    },
    inputs = [],
    outputs = ["//command_line_option:java_language_version"],
)

def _transition_java_language_8_files_impl(ctx):
    return [
        DefaultInfo(
            files = depset(ctx.files.files),
        ),
    ]

_transitioned_java_8_files = rule(
    implementation = _transition_java_language_8_files_impl,
    attrs = {
        "files": attr.label_list(
            allow_files = True,
            cfg = _java_language_version_8_transition,
            mandatory = True,
        ),
    },
)

def transition_java_language_8_filegroup(name, files, visibility):
    _transitioned_java_8_files(
        name = name,
        files = files,
        visibility = visibility,
    )
