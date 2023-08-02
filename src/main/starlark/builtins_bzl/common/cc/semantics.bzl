# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Semantics for Bazel cc rules"""

load(":common/cc/cc_helper.bzl", "cc_helper")

def _get_proto_aspects():
    return []

def _should_create_empty_archive():
    return False

def _validate_deps(ctx):
    pass

def _validate_attributes(ctx):
    pass

def _determine_headers_checking_mode(ctx):
    return "strict"

def _get_semantics():
    return _builtins.internal.bazel_cc_internal.semantics

def _get_stl():
    return attr.label()

def _get_repo():
    return "bazel_tools"

def _get_platforms_root():
    return "platforms//"

def _additional_fragments():
    return []

def _get_distribs_attr():
    return {}

def _get_licenses_attr():
    # TODO(b/182226065): Change to applicable_licenses
    return {}

def _get_loose_mode_in_hdrs_check_allowed_attr():
    return {}

def _def_parser_computed_default(name, tags):
    # This is needed to break the dependency cycle.
    if "__DONT_DEPEND_ON_DEF_PARSER__" in tags or "def_parser" in name:
        return None
    else:
        return Label("@bazel_tools//tools/def_parser:def_parser")

def _get_def_parser():
    return attr.label(
        default = _def_parser_computed_default,
        allow_single_file = True,
        cfg = "exec",
    )

def _get_grep_includes():
    return attr.label()

def _get_runtimes_toolchain():
    return []

def _get_test_malloc_attr():
    return {}

def _get_coverage_attrs():
    return {
        "_lcov_merger": attr.label(
            default = configuration_field(fragment = "coverage", name = "output_generator"),
            executable = True,
            cfg = "exec",
        ),
        "_collect_cc_coverage": attr.label(
            default = "@bazel_tools//tools/test:collect_cc_coverage",
            executable = True,
            cfg = "exec",
        ),
    }

def _get_coverage_env(ctx):
    runfiles = ctx.runfiles()
    test_env = {}
    if ctx.configuration.coverage_enabled:
        # Bazel’s coverage runner
        # (https://github.com/bazelbuild/bazel/blob/3.0.0/tools/test/collect_coverage.sh)
        # needs a binary called “lcov_merge.”  Its location is passed in the
        # LCOV_MERGER environmental variable.  For builtin rules, this variable
        # is set automatically based on a magic “$lcov_merger” or
        # “:lcov_merger” attribute, but it’s not possible to create such
        # attributes in Starlark.  Therefore we specify the variable ourselves.
        # Note that the coverage runner runs in the runfiles root instead of
        # the execution root, therefore we use “path” instead of “short_path.”
        test_env["LCOV_MERGER"] = ctx.executable._lcov_merger.path

        # C/C++ coverage instrumentation needs another binary that wraps gcov;
        # see
        # https://github.com/bazelbuild/bazel/blob/5.0.0/tools/test/collect_coverage.sh#L199.
        # This is normally set from a hidden “$collect_cc_coverage” attribute;
        # see
        # https://github.com/bazelbuild/bazel/blob/5.0.0/src/main/java/com/google/devtools/build/lib/analysis/test/TestActionBuilder.java#L253-L258.
        test_env["CC_CODE_COVERAGE_SCRIPT"] = ctx.executable._collect_cc_coverage.path

        # The test runfiles need all applicable runfiles for the tools above.
        runfiles = runfiles.merge_all([
            ctx.attr._lcov_merger[DefaultInfo].default_runfiles,
            ctx.attr._collect_cc_coverage[DefaultInfo].default_runfiles,
        ])

    return runfiles, test_env

def _get_cc_runtimes(ctx, is_library):
    if is_library:
        return []

    runtimes = [ctx.attr.link_extra_lib]

    if ctx.fragments.cpp.custom_malloc != None:
        runtimes.append(ctx.attr._default_malloc)
    else:
        runtimes.append(ctx.attr.malloc)

    return runtimes

def _get_implementation_deps_allowed_attr():
    return {}

def _check_can_use_implementation_deps(ctx):
    experimental_cc_implementation_deps = ctx.fragments.cpp.experimental_cc_implementation_deps()
    if (not experimental_cc_implementation_deps and ctx.attr.implementation_deps):
        fail("requires --experimental_cc_implementation_deps", attr = "implementation_deps")

def _get_linkstatic_default(ctx):
    if ctx.attr._is_test:
        # By default Tests do not link statically. Except on Windows.
        if cc_helper.has_target_constraints(ctx, ctx.attr._windows_constraints):
            return True
        else:
            return False
    else:
        # Binaries link statically.
        return True

def _get_nocopts_attr():
    return {}

def _get_experimental_link_static_libraries_once(ctx):
    return ctx.fragments.cpp.experimental_link_static_libraries_once()

def _check_cc_shared_library_tags(ctx):
    pass

semantics = struct(
    ALLOWED_RULES_IN_DEPS = [
        "cc_library",
        "objc_library",
        "cc_proto_library",
        "cc_import",
    ],
    ALLOWED_FILES_IN_DEPS = [
        ".ld",
        ".lds",
        ".ldscript",
    ],
    ALLOWED_RULES_WITH_WARNINGS_IN_DEPS = [],
    validate_deps = _validate_deps,
    validate_attributes = _validate_attributes,
    determine_headers_checking_mode = _determine_headers_checking_mode,
    get_semantics = _get_semantics,
    get_repo = _get_repo,
    get_platforms_root = _get_platforms_root,
    additional_fragments = _additional_fragments,
    get_distribs_attr = _get_distribs_attr,
    get_licenses_attr = _get_licenses_attr,
    get_loose_mode_in_hdrs_check_allowed_attr = _get_loose_mode_in_hdrs_check_allowed_attr,
    get_def_parser = _get_def_parser,
    get_stl = _get_stl,
    should_create_empty_archive = _should_create_empty_archive,
    get_grep_includes = _get_grep_includes,
    get_implementation_deps_allowed_attr = _get_implementation_deps_allowed_attr,
    check_can_use_implementation_deps = _check_can_use_implementation_deps,
    get_linkstatic_default = _get_linkstatic_default,
    get_runtimes_toolchain = _get_runtimes_toolchain,
    get_test_malloc_attr = _get_test_malloc_attr,
    get_cc_runtimes = _get_cc_runtimes,
    get_coverage_attrs = _get_coverage_attrs,
    get_coverage_env = _get_coverage_env,
    get_proto_aspects = _get_proto_aspects,
    get_nocopts_attr = _get_nocopts_attr,
    get_experimental_link_static_libraries_once = _get_experimental_link_static_libraries_once,
    check_cc_shared_library_tags = _check_cc_shared_library_tags,
)
