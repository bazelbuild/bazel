# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""Various things common to rules."""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(
    ":common/python/providers.bzl",
    "PyCcLinkParamsProvider",
    "PyInfo",
)
load(":common/python/semantics.bzl", "IMPORTS_ATTR_SUPPORTED", "PyWrapCcInfo")

py_builtins = _builtins.internal.py_builtins
platform_common = _builtins.toplevel.platform_common
CcInfo = _builtins.toplevel.CcInfo
cc_common = _builtins.toplevel.cc_common
coverage_common = _builtins.toplevel.coverage_common

# Extensions without the dot
PYTHON_SOURCE_EXTENSIONS = ["py"]

def union_attrs(*attr_dicts, allow_none = False):
    """Helper for combining and building attriute dicts for rules.

    Similar to dict.update, except:
      * Duplicate keys raise an error if they aren't equal. This is to prevent
        unintentionally replacing an attribute with a potentially incompatible
        definition.
      * None values are special: They mean the attribute is required, but the
        value should be provided by another attribute dict (depending on the
        `allow_none` arg).
    Args:
        *attr_dicts: The dicts to combine.
        allow_none: bool, if True, then None values are allowed. If False,
            then one of `attrs_dicts` must set a non-None value for keys
            with a None value.

    Returns:
        dict of attributes.
    """
    result = {}
    missing = {}
    for attr_dict in attr_dicts:
        for attr_name, value in attr_dict.items():
            if value == None and not allow_none:
                if attr_name not in result:
                    missing[attr_name] = None
            else:
                if attr_name in missing:
                    missing.pop(attr_name)

                if attr_name not in result or result[attr_name] == None:
                    result[attr_name] = value
                elif value != None and result[attr_name] != value:
                    fail("Duplicate attribute name: '{}': existing={}, new={}".format(
                        attr_name,
                        result[attr_name],
                        value,
                    ))

                # Else, they're equal, so do nothing. This allows merging dicts
                # that both define the same key from a common place.

    if missing and not allow_none:
        fail("Required attributes missing: " + csv(missing.keys()))
    return result

def csv(values):
    """Convert a list of strings to comma separated value string."""
    return ", ".join(sorted(values))

def filter_to_py_srcs(srcs):
    """Filters .py files from the given list of files"""

    # TODO(b/203567235): Get the set of recognized extensions from
    # elsewhere, as there may be others. e.g. Bazel recognizes .py3
    # as a valid extension.
    return [f for f in srcs if f.extension == "py"]

def collect_cc_info(ctx, extra_deps = []):
    """Collect the CcInfos from deps

    Args:
        ctx: rule context
        extra_deps: list of additional targets to include

    Returns:
        Merged CcInfo from the targets.
    """
    deps = ctx.attr.deps
    if extra_deps:
        deps = list(deps)
        deps.extend(extra_deps)
    return collect_cc_info_from(deps)

def collect_cc_info_from(deps):
    """Collect the CcInfos from deps

    Args:
        deps: (list[Target]) list of all targets to include

    Returns:
        (CcInfo) Merged CcInfo from the targets.
    """
    cc_infos = []
    for dep in deps:
        if CcInfo in dep:
            cc_infos.append(dep[CcInfo])
        elif PyCcLinkParamsProvider in dep:
            cc_infos.append(dep[PyCcLinkParamsProvider].cc_info)
        elif PyWrapCcInfo and PyWrapCcInfo in dep:
            # TODO(b/203567235): Google specific
            cc_infos.append(dep[PyWrapCcInfo].cc_info)

    return cc_common.merge_cc_infos(cc_infos = cc_infos)

def collect_runfiles(ctx, files):
    """Collects the necessary files from the rule's context.

    This presumes the ctx is for a py_binary, py_test, or py_library rule.

    Args:
        ctx: rule ctx
        files: depset of extra files to include in the runfiles.
    Returns:
        runfiles necessary for the ctx's target.
    """
    return ctx.runfiles(
        transitive_files = files,
        # This little arg carries a lot of weight, but because Starlark doesn't
        # have a way to identify if a target is just a File, the equivalent
        # logic can't be re-implemented in pure-Starlark.
        #
        # Under the hood, it calls the Java `Runfiles#addRunfiles(ctx,
        # DEFAULT_RUNFILES)` method, which is the what the Java implementation
        # of the Python rules originally did, and the details of how that method
        # works have become relied on in various ways. Specifically, what it
        # does is visit the srcs, deps, and data attributes in the following
        # ways:
        #
        # For each target in the "data" attribute...
        #   If the target is a File, then add that file to the runfiles.
        #   Otherwise, add the target's **data runfiles** to the runfiles.
        #
        # Note that, contray to best practice, the default outputs of the
        # targets in `data` are *not* added, nor are the default runfiles.
        #
        # This ends up being important for several reasons, some of which are
        # specific to Google-internal features of the rules.
        #   * For Python executables, we have to use `data_runfiles` to avoid
        #     conflicts for the build data files. Such files have
        #     target-specific content, but uses a fixed location, so if a
        #     binary has another binary in `data`, and both try to specify a
        #     file for that file path, then a warning is printed and an
        #     arbitrary one will be used.
        #   * For rules with _entirely_ different sets of files in data runfiles
        #     vs default runfiles vs default outputs. For example,
        #     proto_library: documented behavior of this rule is that putting it
        #     in the `data` attribute will cause the transitive closure of
        #     `.proto` source files to be included. This set of sources is only
        #     in the `data_runfiles` (`default_runfiles` is empty).
        #   * For rules with a _subset_ of files in data runfiles. For example,
        #     a certain Google rule used for packaging arbitrary binaries will
        #     generate multiple versions of a binary (e.g. different archs,
        #     stripped vs un-stripped, etc) in its default outputs, but only
        #     one of them in the runfiles; this helps avoid large, unused
        #     binaries contributing to remote executor input limits.
        #
        # Unfortunately, the above behavior also results in surprising behavior
        # in some cases. For example, simple custom rules that only return their
        # files in their default outputs won't have their files included. Such
        # cases must either return their files in runfiles, or use `filegroup()`
        # which will do so for them.
        #
        # For each target in "srcs" and "deps"...
        #   Add the default runfiles of the target to the runfiles. While this
        #   is desirable behavior, it also ends up letting a `py_library`
        #   be put in `srcs` and still mostly work.
        # TODO(b/224640180): Reject py_library et al rules in srcs.
        collect_default = True,
    )

def create_py_info(ctx, direct_sources):
    """Create PyInfo provider.

    Args:
        ctx: rule ctx.
        direct_sources: depset of Files; the direct, raw `.py` sources for the
            target. This should only be Python source files. It should not
            include pyc files.

    Returns:
        A tuple of the PyInfo instance and a depset of the
        transitive sources collected from dependencies (the latter is only
        necessary for deprecated extra actions support).
    """
    uses_shared_libraries = False
    transitive_sources_depsets = []  # list of depsets
    transitive_sources_files = []  # list of Files
    for target in ctx.attr.deps:
        # PyInfo may not be present for e.g. cc_library rules.
        if PyInfo in target:
            info = target[PyInfo]
            transitive_sources_depsets.append(info.transitive_sources)
            uses_shared_libraries = uses_shared_libraries or info.uses_shared_libraries
        else:
            # TODO(b/228692666): Remove this once non-PyInfo targets are no
            # longer supported in `deps`.
            files = target.files.to_list()
            for f in files:
                if f.extension == "py":
                    transitive_sources_files.append(f)
                uses_shared_libraries = (
                    uses_shared_libraries or
                    cc_helper.is_valid_shared_library_artifact(f)
                )
    deps_transitive_sources = depset(
        direct = transitive_sources_files,
        transitive = transitive_sources_depsets,
    )

    # We only look at data to calculate uses_shared_libraries, if it's already
    # true, then we don't need to waste time looping over it.
    if not uses_shared_libraries:
        # Similar to the above, except we only calculate uses_shared_libraries
        for target in ctx.attr.data:
            # TODO(b/234730058): Remove checking for PyInfo in data once depot
            # cleaned up.
            if PyInfo in target:
                info = target[PyInfo]
                uses_shared_libraries = info.uses_shared_libraries
            else:
                files = target.files.to_list()
                for f in files:
                    uses_shared_libraries = cc_helper.is_valid_shared_library_artifact(f)
                    if uses_shared_libraries:
                        break
            if uses_shared_libraries:
                break

    # TODO(b/203567235): Set `uses_shared_libraries` field, though the Bazel
    # docs indicate it's unused in Bazel and may be removed.
    py_info = PyInfo(
        transitive_sources = depset(
            transitive = [deps_transitive_sources, direct_sources],
        ),
        # TODO(b/203567235): Implement imports attribute
        imports = depset() if IMPORTS_ATTR_SUPPORTED else depset(),
        # NOTE: This isn't strictly correct, but with Python 2 gone,
        # the srcs_version logic is largely defunct, so shouldn't matter in
        # practice.
        has_py2_only_sources = False,
        has_py3_only_sources = False,
        uses_shared_libraries = uses_shared_libraries,
    )
    return py_info, deps_transitive_sources

def create_instrumented_files_info(ctx):
    return coverage_common.instrumented_files_info(
        ctx,
        source_attributes = ["srcs"],
        dependency_attributes = ["deps", "data"],
        extensions = PYTHON_SOURCE_EXTENSIONS,
    )

def create_output_group_info(transitive_sources):
    return OutputGroupInfo(
        compilation_prerequisites_INTERNAL_ = transitive_sources,
        compilation_outputs = transitive_sources,
    )

_BOOL_TYPE = type(True)

def is_bool(v):
    return type(v) == _BOOL_TYPE

def target_platform_has_any_constraint(ctx, constraints):
    """Check if target platform has any of a list of constraints.

    Args:
      ctx: rule context.
      constraints: label_list of constraints.

    Returns:
      True if target platform has at least one of the constraints.
    """
    for constraint in constraints:
        constraint_value = constraint[platform_common.ConstraintValueInfo]
        if ctx.target_platform_has_constraint(constraint_value):
            return True
    return False
