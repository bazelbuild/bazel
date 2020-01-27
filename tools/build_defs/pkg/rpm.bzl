# Copyright 2017 The Bazel Authors. All rights reserved.
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
"""Rules to create RPM archives."""

load("//tools/config:common_settings.bzl", "BuildSettingInfo")

rpm_filetype = [".rpm"]

spec_filetype = [".spec"]

def _pkg_rpm_impl(ctx):
    """Implements to pkg_rpm rule."""

    if ctx.attr._no_build_defs_pkg_flag[BuildSettingInfo].value:
        fail("The built-in version of pkg_rpm has been removed. Please use" +
             " https://github.com/bazelbuild/rules_pkg/blob/master/pkg.")

    files = []
    args = ["--name=" + ctx.label.name]
    if ctx.attr.rpmbuild_path:
        args += ["--rpmbuild=" + ctx.attr.rpmbuild_path]

    # Version can be specified by a file or inlined.
    if ctx.attr.version_file:
        if ctx.attr.version:
            fail("Both version and version_file attributes were specified")
        args += ["--version=@" + ctx.file.version_file.path]
        files += [ctx.file.version_file]
    elif ctx.attr.version:
        args += ["--version=" + ctx.attr.version]

    # Release can be specified by a file or inlined.
    if ctx.attr.release_file:
        if ctx.attr.release:
            fail("Both release and release_file attributes were specified")
        args += ["--release=@" + ctx.file.release_file.path]
        files += [ctx.file.release_file]
    elif ctx.attr.release:
        args += ["--release=" + ctx.attr.release]

    if ctx.attr.architecture:
        args += ["--arch=" + ctx.attr.architecture]

    if not ctx.attr.spec_file:
        fail("spec_file was not specified")

    # Expand the spec file template.
    spec_file = ctx.actions.declare_file("%s.spec" % ctx.label.name)

    # Create the default substitutions based on the data files.
    substitutions = {}
    for data_file in ctx.files.data:
        key = "{%s}" % data_file.basename
        substitutions[key] = data_file.path
    ctx.actions.expand_template(
        template = ctx.file.spec_file,
        output = spec_file,
        substitutions = substitutions,
    )
    args += ["--spec_file=" + spec_file.path]
    files += [spec_file]

    args += ["--out_file=" + ctx.outputs.rpm.path]

    # Add data files.
    if ctx.file.changelog:
        files += [ctx.file.changelog]
        args += [ctx.file.changelog.path]
    files += ctx.files.data

    for f in ctx.files.data:
        args += [f.path]

    if ctx.attr.debug:
        args += ["--debug"]

    # Call the generator script.
    # TODO(katre): Generate a source RPM.
    ctx.actions.run(
        executable = ctx.executable._make_rpm,
        use_default_shell_env = True,
        arguments = args,
        inputs = files,
        outputs = [ctx.outputs.rpm],
        mnemonic = "MakeRpm",
    )

    # Link the RPM to the expected output name.
    ctx.actions.run(
        executable = "ln",
        arguments = [
            "-s",
            ctx.outputs.rpm.basename,
            ctx.outputs.out.path,
        ],
        inputs = [ctx.outputs.rpm],
        outputs = [ctx.outputs.out],
    )

    # Link the RPM to the RPM-recommended output name.
    if "rpm_nvra" in dir(ctx.outputs):
        ctx.actions.run(
            executable = "ln",
            arguments = [
                "-s",
                ctx.outputs.rpm.basename,
                ctx.outputs.rpm_nvra.path,
            ],
            inputs = [ctx.outputs.rpm],
            outputs = [ctx.outputs.rpm_nvra],
        )

def _pkg_rpm_outputs(version, release):
    outputs = {
        "out": "%{name}.rpm",
        "rpm": "%{name}-%{architecture}.rpm",
    }

    # The "rpm_nvra" output follows the recommended package naming convention of
    # Name-Version-Release.Arch.rpm
    # See http://ftp.rpm.org/max-rpm/ch-rpm-file-format.html
    if version and release:
        outputs["rpm_nvra"] = "%{name}-%{version}-%{release}.%{architecture}.rpm"

    return outputs

# Define the rule.
pkg_rpm = rule(
    attrs = {
        "spec_file": attr.label(
            mandatory = True,
            allow_single_file = spec_filetype,
        ),
        "architecture": attr.string(default = "all"),
        "version_file": attr.label(
            allow_single_file = True,
        ),
        "version": attr.string(),
        "changelog": attr.label(
            allow_single_file = True,
        ),
        "data": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "release_file": attr.label(allow_single_file = True),
        "release": attr.string(),
        "debug": attr.bool(default = False),
        # Implicit dependencies.
        "rpmbuild_path": attr.string(),
        "_make_rpm": attr.label(
            default = Label("//tools/build_defs/pkg:make_rpm"),
            cfg = "host",
            executable = True,
            allow_files = True,
        ),
        "_no_build_defs_pkg_flag": attr.label(
            default = "//tools/build_defs/pkg:incompatible_no_build_defs_pkg",
        ),
    },
    executable = False,
    outputs = _pkg_rpm_outputs,
    implementation = _pkg_rpm_impl,
)

"""Creates an RPM format package from the data files.

This runs rpmbuild (and requires it to be installed beforehand) to generate
an RPM package based on the spec_file and data attributes.

Two outputs are guaranteed to be produced: "%{name}.rpm", and
"%{name}-%{architecture}.rpm". If the "version" and "release" arguments are
non-empty, a third output will be produced, following the RPM-recommended
N-V-R.A format (Name-Version-Release.Architecture.rpm). Note that due to
the fact that rule implementations cannot access the contents of files,
the "version_file" and "release_file" arguments will not create an output
using N-V-R.A format.

Args:
  spec_file: The RPM spec file to use. If the version or version_file
    attributes are provided, the Version in the spec will be overwritten,
    and likewise behaviour with release and release_file. Any Sources listed
    in the spec file must be provided as data dependencies.
    The base names of data dependencies can be replaced with the actual location
    using "{basename}" syntax.
  version: The version of the package to generate. This will overwrite any
    Version provided in the spec file. Only specify one of version and
    version_file.
  version_file: A file containing the version of the package to generate. This
    will overwrite any Version provided in the spec file. Only specify one of
    version and version_file.
  release: The release of the package to generate. This will overwrite any
    release provided in the spec file. Only specify one of release and
    release_file.
  release_file: A file containing the release of the package to generate. This
    will overwrite any release provided in the spec file. Only specify one of
    release and release_file.
  changelog: A changelog file to include. This will not be written to the spec
    file, which should only list changes to the packaging, not the software itself.
  data: List all files to be included in the package here.
"""
