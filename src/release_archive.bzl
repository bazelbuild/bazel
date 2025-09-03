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

"""Rules to create a release archive"""

load("@rules_java//java:java_binary.bzl", "java_binary")
load("@with_cfg.bzl", "with_cfg")

# The minimum --java_{tool_,}runtime_version supported by prebuilt Java tools.
_MINIMUM_JAVA_RUNTIME_VERSION = 8

# The minimum version of a java_toolchain's java_runtime supported by prebuilt Java tools.
_MINIMUM_JAVA_COMPILATION_RUNTIME_VERSION = 17

minimum_java_runtime_java_binary, _minimum_java_runtime_java_binary = (
    # Don't warn about targeting very old Java versions.
    with_cfg(java_binary)
        .set("java_language_version", str(_MINIMUM_JAVA_RUNTIME_VERSION))
        .extend("javacopt", ["-Xlint:-options"])
        .build()
)

minimum_java_runtime_filegroup, _minimum_java_runtime_filegroup = (
    # Don't warn about targeting very old Java versions.
    with_cfg(native.filegroup)
        .set("java_language_version", str(_MINIMUM_JAVA_RUNTIME_VERSION))
        .extend("javacopt", ["-Xlint:-options"])
        .build()
)

minimum_java_compilation_runtime_filegroup, _minimum_java_compilation_runtime_filegroup = (
    with_cfg(native.filegroup)
        .set("java_language_version", str(_MINIMUM_JAVA_COMPILATION_RUNTIME_VERSION))
        .build()
)

def release_archive(name, srcs = [], src_map = {}, package_dir = "-", deps = [], **kwargs):
    """ Creates an zip of the srcs, and renamed label artifacts.

    Usage:
    //:BUILD
    load("//src:release_archive.bzl", "release_archive")
    release_archive(
        name = "release_archive",
        src_map = {
            "BUILD.release.bazel.bazel": "BUILD.bazel",
            "WORKSPACE.release.bazel": "WORKSPACE",
        },
        deps = [
            "//dep:pkg"
        ],
    )
    //dep:BUILD
    load("//src:release_archive.bzl", "release_archive")
    release_archive(
        name = "pkg",
        srcs = [
            ":label_of_artifact",
        ],
    )
    Args:
        name: target identifier, points to a pkg_tar target.
        package_dir: directory to place the srcs, src_map, and dist_files under. Defaults to the current directory.
        src_map: dict of <label>:<name string> for labels to be renamed and included in the distribution.
        srcs: files to include in the distribution.
        deps: release_archives to be included.
        **kwargs: other arguments added to the final genrule (for example visibility)
    """
    srcs = list(srcs)
    for source, target in src_map.items():
        rename_name = name + "_" + target
        _rename(
            name = rename_name,
            source = source,
            target = target,
        )
        srcs.append(rename_name)

    if srcs != []:
        native.genrule(
            name = name + "_srcs",
            srcs = srcs,
            outs = [name + "_srcs.zip"],
            cmd = "zip -qjX $@ $$(echo $(SRCS) | sort)",
            visibility = ["//visibility:private"],
        )
        deps = [name + "_srcs.zip"] + deps

    native.genrule(
        name = name,
        srcs = deps,
        outs = [(name[:-len("_zip")] if name.endswith("_zip") else name) + ".zip"],
        cmd = "$(location //src:merge_zip_files) %s $@ $(SRCS)" % package_dir,
        output_to_bindir = 1,
        tools = ["//src:merge_zip_files"],
        **kwargs
    )

def _rename_impl(ctx):
    out_file = ctx.actions.declare_file(ctx.label.name + "/" + ctx.attr.target)
    in_file = ctx.file.source
    ctx.actions.run_shell(
        inputs = [in_file],
        outputs = [out_file],
        progress_message = "%s -> %s" % (in_file, ctx.attr.target),
        command = "mkdir -p {dir} && cp {in_file} {out_file}".format(
            dir = out_file.dirname,
            in_file = in_file.path,
            out_file = out_file.path,
        ),
    )
    return [DefaultInfo(files = depset([out_file]))]

_rename = rule(
    implementation = _rename_impl,
    attrs = {
        "source": attr.label(allow_single_file = True, mandatory = True),
        "target": attr.string(mandatory = True),
    },
)
