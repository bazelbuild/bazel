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

"""Starlark implementation of cc_import.

We may change the implementation at any moment or even delete this file. Do not
rely on this. Pass the flag --experimental_starlark_cc_import
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _to_list(element):
    if element == None:
        return []
    else:
        return [element]

def _to_depset(element):
    if element == None:
        return depset()
    else:
        return depset([element])

def _is_shared_library_extension_valid(shared_library_name):
    if (shared_library_name.endswith(".so") or
        shared_library_name.endswith(".dll") or
        shared_library_name.endswith(".dylib")):
        return True

    # validate against the regex "^.+\.so(\.\d\w*)+$" for versioned .so files
    parts = shared_library_name.split(".")
    extension = parts[1]
    if extension != "so":
        return False
    version_parts = parts[2:]
    for part in version_parts:
        if not part[0].isdigit():
            return False
        for c in part[1:].elems():
            if not (c.isalnum() or c == "_"):
                return False
    return True

def _perform_error_checks(
        system_provided,
        shared_library_artifact,
        interface_library_artifact):
    # If the shared library will be provided by system during runtime, users are not supposed to
    # specify shared_library.
    if system_provided and shared_library_artifact != None:
        fail("'shared_library' shouldn't be specified when 'system_provided' is true")

    # If a shared library won't be provided by system during runtime and we are linking the shared
    # library through interface library, the shared library must be specified.
    if (not system_provided and shared_library_artifact == None and
        interface_library_artifact != None):
        fail("'shared_library' should be specified when 'system_provided' is false")

    if (shared_library_artifact != None and
        not _is_shared_library_extension_valid(shared_library_artifact.basename)):
        fail("'shared_library' does not produce any cc_import shared_library files (expected .so, .dylib or .dll)")

def _get_no_pic_and_pic_static_library(static_library):
    if static_library == None:
        return (None, None)

    if static_library.extension == ".pic.a":
        return (None, static_library)
    else:
        return (static_library, None)

def _cc_import_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_common.check_experimental_starlark_cc_import(
        actions = ctx.actions,
    )

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    _perform_error_checks(
        ctx.attr.system_provided,
        ctx.file.shared_library,
        ctx.file.interface_library,
    )

    (no_pic_static_library, pic_static_library) = _get_no_pic_and_pic_static_library(
        ctx.file.static_library,
    )

    library_to_link = cc_common.create_library_to_link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo],
        static_library = no_pic_static_library,
        pic_static_library = pic_static_library,
        dynamic_library = ctx.file.shared_library,
        interface_library = ctx.file.interface_library,
        alwayslink = ctx.attr.alwayslink,
    )

    linking_context = cc_common.create_linking_context(
        libraries_to_link = [library_to_link],
        user_link_flags = ctx.attr.linkopts,
    )

    (compilation_context, compilation_outputs) = cc_common.compile(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        public_hdrs = ctx.files.hdrs,
        includes = ctx.attr.includes,
        name = ctx.label.name,
    )

    return [CcInfo(
        compilation_context = compilation_context,
        linking_context = linking_context,
    )]

cc_import = rule(
    implementation = _cc_import_impl,
    attrs = {
        "hdrs": attr.label_list(allow_files = [".h"]),
        "static_library": attr.label(allow_single_file = [".a", ".lib", ".pic.a"]),
        "shared_library": attr.label(allow_single_file = True),
        "interface_library": attr.label(
            allow_single_file = [".ifso", ".tbd", ".lib", ".so", ".dylib"],
        ),
        "system_provided": attr.bool(default = False),
        "alwayslink": attr.bool(default = False),
        "linkopts": attr.string_list(),
        "includes": attr.string_list(),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    toolchains = ["@rules_cc//cc:toolchain_type"],  # copybara-use-repo-external-label
    fragments = ["cpp"],
)
