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

java_common = _builtins.toplevel.java_common
JavaPluginInfo = _builtins.toplevel.JavaPluginInfo
JavaInfo = _builtins.toplevel.JavaInfo

def _macro_preprocess(kwargs):
    pass

def _check_rule(ctx):
    pass

def _check_dependency_rule_kinds(ctx, rule_class = ""):
    pass

def _preprocess(ctx):
    return []

def _postprocess(ctx, base_info):
    return base_info.java_info

def _postprocess_plugin(ctx, base_info):
    return base_info.java_info, base_info.default_info

def _check_proto_registry_collision(ctx):
    pass

def _get_coverage_runner(ctx):
    runner = ctx.attr._java_toolchain[java_common.JavaToolchainInfo].jacocorunner
    if not runner:
        fail("jacocorunner not set in java_toolchain")
    runner_jar = runner.executable
    # wrap the jar in JavaInfo so we can add it to deps for java_common.compile()
    return JavaInfo(output_jar = runner_jar, compile_jar = runner_jar)

semantics = struct(
    COLLECT_SRCS_FROM_PROTO_LIBRARY = False,
    JAVA_TOOLCHAIN_LABEL = "@bazel_tools//tools/jdk:current_java_toolchain",
    JAVA_PLUGINS_FLAG_ALIAS_LABEL = "@bazel_tools//tools/jdk:java_plugins_flag_alias",
    PROGUARD_ALLOWLISTER_LABEL = "@bazel_tools//tools/jdk:proguard_whitelister",
    EXTRA_SRCS_TYPES = [],
    EXTRA_ATTRIBUTES = {
        "resource_strip_prefix": attr.string(),
    },
    EXTRA_PLUGIN_ATTRIBUTES = {
        "resource_strip_prefix": attr.string(),
    },
    EXTRA_DEPS = [],
    EXTRA_PLUGIN_DEPS = [],
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
    macro_preprocess = _macro_preprocess,
    check_rule = _check_rule,
    check_dependency_rule_kinds = _check_dependency_rule_kinds,
    preprocess = _preprocess,
    postprocess = _postprocess,
    postprocess_plugin = _postprocess_plugin,
    check_proto_registry_collision = _check_proto_registry_collision,
    get_coverage_runner = _get_coverage_runner,
)
