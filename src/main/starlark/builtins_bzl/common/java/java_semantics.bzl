# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""
Java Semantics
"""

def _postprocess(ctx, base_info):
    return base_info.java_info

def _find_java_toolchain(ctx):
    return ctx.toolchains["@bazel_tools//tools/jdk:toolchain_type"].java

def _find_java_runtime_toolchain(ctx):
    return ctx.toolchains["@bazel_tools//tools/jdk:runtime_toolchain_type"].java_runtime

def _get_default_resource_path(path, segment_extractor):
    # Look for src/.../resources to match Maven repository structure.
    segments = path.split("/")
    for idx in range(0, len(segments) - 2):
        if segments[idx] == "src" and segments[idx + 2] == "resources":
            return "/".join(segments[idx + 3:])
    java_segments = segment_extractor(path)
    return "/".join(java_segments) if java_segments != None else path

def _compatible_javac_options(*_args):
    return depset()

def _check_java_info_opens_exports():
    pass

def _minimize_cc_info(cc_info):
    return cc_info

_DOCS = struct(
    ATTRS = {
        "resources": """
<p>
If resources are specified, they will be bundled in the jar along with the usual
<code>.class</code> files produced by compilation. The location of the resources inside
of the jar file is determined by the project structure. Bazel first looks for Maven's
<a href="https://maven.apache.org/guides/introduction/introduction-to-the-standard-directory-layout.html">standard directory layout</a>,
(a "src" directory followed by a "resources" directory grandchild). If that is not
found, Bazel then looks for the topmost directory named "java" or "javatests" (so, for
example, if a resource is at <code>&lt;workspace root&gt;/x/java/y/java/z</code>, the
path of the resource will be <code>y/java/z</code>. This heuristic cannot be overridden,
however, the <code>resource_strip_prefix</code> attribute can be used to specify a
specific alternative directory for resource files.
    """,
        "use_testrunner": """
Use the test runner (by default
<code>com.google.testing.junit.runner.BazelTestRunner</code>) class as the
main entry point for a Java program, and provide the test class
to the test runner as a value of <code>bazel.test_suite</code>
system property.
    """,
    },
)

semantics = struct(
    JAVA_TOOLCHAIN_LABEL = "@bazel_tools//tools/jdk:current_java_toolchain",
    JAVA_TOOLCHAIN_TYPE = "@bazel_tools//tools/jdk:toolchain_type",
    JAVA_TOOLCHAIN = _builtins.toplevel.config_common.toolchain_type("@bazel_tools//tools/jdk:toolchain_type", mandatory = True),
    find_java_toolchain = _find_java_toolchain,
    JAVA_RUNTIME_TOOLCHAIN_TYPE = "@bazel_tools//tools/jdk:runtime_toolchain_type",
    JAVA_RUNTIME_TOOLCHAIN = _builtins.toplevel.config_common.toolchain_type("@bazel_tools//tools/jdk:runtime_toolchain_type", mandatory = True),
    find_java_runtime_toolchain = _find_java_runtime_toolchain,
    JAVA_PLUGINS_FLAG_ALIAS_LABEL = "@bazel_tools//tools/jdk:java_plugins_flag_alias",
    EXTRA_SRCS_TYPES = [],
    ALLOWED_RULES_IN_DEPS = [
        "cc_binary",  # NB: linkshared=1
        "cc_library",
        "genrule",
        "genproto",  # TODO(bazel-team): we should filter using providers instead (starlark rule).
        "java_import",
        "java_library",
        "java_proto_library",
        "java_lite_proto_library",
        "proto_library",
        "sh_binary",
        "sh_library",
    ],
    ALLOWED_RULES_IN_DEPS_WITH_WARNING = [],
    LINT_PROGRESS_MESSAGE = "Running Android Lint for: %{label}",
    JAVA_STUB_TEMPLATE_LABEL = "@bazel_tools//tools/jdk:java_stub_template.txt",
    BUILD_INFO_TRANSLATOR_LABEL = "@bazel_tools//tools/build_defs/build_info:java_build_info",
    JAVA_TEST_RUNNER_LABEL = "@bazel_tools//tools/jdk:TestRunner",
    IS_BAZEL = True,
    get_default_resource_path = _get_default_resource_path,
    compatible_javac_options = _compatible_javac_options,
    LAUNCHER_FLAG_LABEL = Label("@bazel_tools//tools/jdk:launcher_flag_alias"),
    JAVA_PROTO_TOOLCHAIN = "@rules_java//java/proto:toolchain_type",
    JAVA_LITE_PROTO_TOOLCHAIN = "@rules_java//java/proto:lite_toolchain_type",
    PROGUARD_ALLOWLISTER_LABEL = "@bazel_tools//tools/jdk:proguard_whitelister",
    check_java_info_opens_exports = _check_java_info_opens_exports,
    DOCS = struct(
        for_attribute = lambda name: _DOCS.ATTRS.get(name, ""),
    ),
    minimize_cc_info = _minimize_cc_info,
)
