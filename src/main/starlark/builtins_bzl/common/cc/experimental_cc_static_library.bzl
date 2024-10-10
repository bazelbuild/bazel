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
"""This is an experimental implementation of cc_static_library.

We may change the implementation at any moment or even delete this file. Do not
rely on this.
"""

load(":common/cc/action_names.bzl", "ACTION_NAMES")
load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "artifact_category", "cc_helper")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/paths.bzl", "paths")

cc_internal = _builtins.internal.cc_internal

def _declare_static_library(*, name, actions, cc_toolchain):
    basename = paths.basename(name)
    new_basename = cc_toolchain.get_artifact_name_for_category(
        category = artifact_category.STATIC_LIBRARY,
        output_name = basename,
    )
    return actions.declare_file(name.removesuffix(basename) + new_basename)

def _collect_linker_inputs(deps):
    transitive_linker_inputs = [dep[CcInfo].linking_context.linker_inputs for dep in deps]
    return depset(transitive = transitive_linker_inputs, order = "topological")

def _flatten_and_get_objects(linker_inputs):
    # Flattening a depset to get the action inputs.
    transitive_objects = []
    for linker_input in linker_inputs.to_list():
        for lib in linker_input.libraries:
            if lib.pic_objects:
                transitive_objects.append(depset(lib.pic_objects))
            elif lib.objects:
                transitive_objects.append(depset(lib.objects))

    return depset(transitive = transitive_objects, order = "topological")

def _archive_objects(*, name, actions, cc_toolchain, feature_configuration, objects):
    static_library = _declare_static_library(
        name = name,
        actions = actions,
        cc_toolchain = cc_toolchain,
    )

    archiver_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
    )
    archiver_variables = cc_common.create_link_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        output_file = static_library.path,
        is_using_linker = False,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
        variables = archiver_variables,
    )
    args = actions.args()
    args.add_all(command_line)
    args.add_all(objects)

    if cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "archive_param_file",
    ):
        # TODO: The flag file arg should come from the toolchain instead.
        args.use_param_file("@%s", use_always = True)

    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
        variables = archiver_variables,
    )
    execution_requirements_keys = cc_common.get_execution_requirements(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
    )

    actions.run(
        executable = archiver_path,
        arguments = [args],
        env = env,
        execution_requirements = {k: "" for k in execution_requirements_keys},
        inputs = depset(transitive = [cc_toolchain.all_files, objects]),
        outputs = [static_library],
        use_default_shell_env = True,
        mnemonic = "CppTransitiveArchive",
        progress_message = "Creating static library %{output}",
    )

    return static_library

def _validate_static_library(*, name, actions, cc_toolchain, feature_configuration, static_library):
    if not cc_common.action_is_enabled(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
    ):
        return None

    validation_output = actions.declare_file(name + "_validation_output.txt")

    validator_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
    )
    args = actions.args()
    args.add(static_library)
    args.add(validation_output)

    execution_requirements_keys = cc_common.get_execution_requirements(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
    )

    actions.run(
        executable = validator_path,
        arguments = [args],
        execution_requirements = {k: "" for k in execution_requirements_keys},
        inputs = depset(
            direct = [static_library],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [validation_output],
        use_default_shell_env = True,
        mnemonic = "ValidateStaticLibrary",
        progress_message = "Validating static library %{label}",
    )

    return validation_output

def _pretty_label(label):
    s = str(label)

    # Emit main repo labels (both with and without --enable_bzlmod) without a
    # repo prefix.
    if s.startswith("@@//") or s.startswith("@//"):
        return s.lstrip("@")
    return s

def _linkdeps_map_each(linker_input):
    has_library = False
    for lib in linker_input.libraries:
        if lib.pic_objects or lib.objects:
            # Has been added to the archive.
            return None
        if lib.pic_static_library != None or lib.static_library != None or lib.dynamic_library != None or lib.interface_library != None:
            has_library = True
    if not has_library:
        # Does not provide any linkable artifact. May still contribute to linkopts.
        return None

    return _pretty_label(linker_input.owner)

def _linkopts_map_each(linker_input):
    return linker_input.user_link_flags

def _format_linker_inputs(*, actions, name, linker_inputs, map_each):
    file = actions.declare_file(name)
    args = actions.args().add_all(linker_inputs, map_each = map_each)
    actions.write(output = file, content = args)
    return file

def _cc_static_library_impl(ctx):
    if not cc_common.check_experimental_cc_static_library():
        fail("cc_static_library is an experimental rule and must be enabled with --experimental_cc_static_library")

    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features + ["symbol_check"],
        unsupported_features = ctx.disabled_features,
    )

    linker_inputs = _collect_linker_inputs(ctx.attr.deps)

    static_library = _archive_objects(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        objects = _flatten_and_get_objects(linker_inputs),
    )

    linkdeps_file = _format_linker_inputs(
        actions = ctx.actions,
        name = ctx.label.name + "_linkdeps.txt",
        linker_inputs = linker_inputs,
        map_each = _linkdeps_map_each,
    )

    linkopts_file = _format_linker_inputs(
        actions = ctx.actions,
        name = ctx.label.name + "_linkopts.txt",
        linker_inputs = linker_inputs,
        map_each = _linkopts_map_each,
    )

    validation_output = _validate_static_library(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        static_library = static_library,
    )

    output_groups = {
        "linkdeps": depset([linkdeps_file]),
        "linkopts": depset([linkopts_file]),
    }
    if validation_output:
        output_groups["_validation"] = depset([validation_output])

    runfiles = ctx.runfiles().merge_all([
        dep[DefaultInfo].default_runfiles
        for dep in ctx.attr.deps
    ])

    return [
        DefaultInfo(
            files = depset([static_library]),
            runfiles = runfiles,
        ),
        OutputGroupInfo(**output_groups),
    ]

cc_static_library = rule(
    implementation = _cc_static_library_impl,
    doc = """
<b>This rule is currently experimental and can only be used with the <code>
--experimental_cc_static_library</code> flag.</b>

Produces a static library from a list of targets and their transitive dependencies.

<p>The resulting static library contains the object files of the targets listed in
<code>deps</code> as well as their transitive dependencies, with preference given to
<code>PIC</code> objects.</p>

<h4 id="cc_static_library_output_groups">Output groups</h4>

<h5><code>linkdeps</code></h5>
<p>A text file containing the labels of those transitive dependencies of targets listed in
<code>deps</code> that did not contribute any object files to the static library, but do
provide at least one static, dynamic or interface library. The resulting static library
may require these libraries to be available at link time.</p>

<h5><code>linkopts</code></h5>
<p>A text file containing the user-provided <code>linkopts</code> of all transitive
dependencies of targets listed in <code>deps</code>.

<h4 id="cc_static_library_symbol_check">Duplicate symbols</h4>
<p>By default, the <code>cc_static_library</code> rule checks that the resulting static
library does not contain any duplicate symbols. If it does, the build fails with an error
message that lists the duplicate symbols and the object files containing them.</p>

<p>This check can be disabled per target or per package by setting
<code>features = ["-symbol_check"]</code> or globally via
<code>--features=-symbol_check</code>.</p>

<h5 id="cc_static_library_symbol_check_toolchain">Toolchain support for <code>symbol_check</code></h5>
<p>The auto-configured C++ toolchains shipped with Bazel support the
<code>symbol_check</code> feature on all platforms. Custom toolchains can add support for
it in one of two ways:</p>
<ul>
  <li>Implementing the <code>ACTION_NAMES.validate_static_library</code> action and
  enabling it with the <code>symbol_check</code> feature. The tool set in the action is
  invoked with two arguments, the static library to check for duplicate symbols and the
  path of a file that must be created if the check passes.</li>
  <li>Having the <code>symbol_check</code> feature add archiver flags that cause the
  action creating the static library to fail on duplicate symbols.</li>
</ul>
""",
    attrs = {
        "deps": attr.label_list(
            providers = [CcInfo],
            doc = """
The list of targets to combine into a static library, including all their transitive
dependencies.

<p>Dependencies that do not provide any object files are not included in the static
library, but their labels are collected in the file provided by the
<code>linkdeps</code> output group.</p>
""",
        ),
    },
    toolchains = cc_helper.use_cpp_toolchain(),
    fragments = ["cpp"],
)
