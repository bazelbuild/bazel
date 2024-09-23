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

config_common = _builtins.toplevel.config_common
coverage_common = _builtins.toplevel.coverage_common
platform_common = _builtins.toplevel.platform_common
RunEnvironmentInfo = _builtins.toplevel.RunEnvironmentInfo

_SH_TOOLCHAIN_TYPE = "@bazel_tools//tools/sh:toolchain_type"

def _sh_executable_impl(ctx):
    if len(ctx.files.srcs) != 1:
        fail("you must specify exactly one file in 'srcs'", attr = "srcs")

    symlink = ctx.actions.declare_file(ctx.label.name)
    src = ctx.files.srcs[0]

    ctx.actions.symlink(
        output = symlink,
        target_file = src,
        is_executable = True,
        progress_message = "Symlinking %{label}",
    )

    direct_files = [src, symlink]

    # TODO: Consider extracting this logic into a function provided by
    # sh_toolchain to allow users to inject launcher creation logic for
    # non-Windows platforms.
    if ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo]):
        main_executable = _launcher_for_windows(ctx, symlink, src)
        direct_files.append(main_executable)
    else:
        main_executable = symlink

    files = depset(direct = direct_files)
    runfiles = ctx.runfiles(transitive_files = files, collect_default = True)
    default_info = DefaultInfo(
        executable = main_executable,
        files = files,
        runfiles = runfiles,
    )

    instrumented_files_info = coverage_common.instrumented_files_info(
        ctx,
        source_attributes = ["srcs"],
        dependency_attributes = ["deps", "data"],
    )

    run_environment_info = RunEnvironmentInfo(
        environment = {
            key: ctx.expand_make_variables(
                "env",
                ctx.expand_location(value, ctx.attr.data, short_paths = True),
                {},
            )
            for key, value in ctx.attr.env.items()
        },
        inherited_environment = ctx.attr.env_inherit,
    )

    return [
        default_info,
        instrumented_files_info,
        run_environment_info,
    ]

_WINDOWS_EXECUTABLE_EXTENSIONS = [
    "exe",
    "cmd",
    "bat",
]

def _is_windows_executable(file):
    return file.extension in _WINDOWS_EXECUTABLE_EXTENSIONS

def _create_windows_exe_launcher(ctx, sh_toolchain, primary_output):
    if not sh_toolchain.launcher or not sh_toolchain.launcher_maker:
        fail("Windows sh_toolchain requires both 'launcher' and 'launcher_maker' to be set")

    bash_launcher = ctx.actions.declare_file(ctx.label.name + ".exe")

    launch_info = ctx.actions.args().use_param_file("%s", use_always = True).set_param_file_format("multiline")
    launch_info.add("binary_type=Bash")
    launch_info.add(ctx.workspace_name, format = "workspace_name=%s")
    launch_info.add("1" if ctx.configuration.runfiles_enabled() else "0", format = "symlink_runfiles_enabled=%s")
    launch_info.add(sh_toolchain.path, format = "bash_bin_path=%s")
    bash_file_short_path = primary_output.short_path
    if bash_file_short_path.startswith("../"):
        bash_file_rlocationpath = bash_file_short_path[3:]
    else:
        bash_file_rlocationpath = ctx.workspace_name + "/" + bash_file_short_path
    launch_info.add(bash_file_rlocationpath, format = "bash_file_rlocationpath=%s")

    launcher_artifact = sh_toolchain.launcher
    ctx.actions.run(
        executable = sh_toolchain.launcher_maker,
        inputs = [launcher_artifact],
        outputs = [bash_launcher],
        arguments = [launcher_artifact.path, launch_info, bash_launcher.path],
        use_default_shell_env = True,
        toolchain = _SH_TOOLCHAIN_TYPE,
    )
    return bash_launcher

def _launcher_for_windows(ctx, primary_output, main_file):
    if _is_windows_executable(main_file):
        if main_file.extension == primary_output.extension:
            return primary_output
        else:
            fail("Source file is a Windows executable file, target name extension should match source file extension")

    # bazel_tools should always registers a toolchain for Windows, but it may have an empty path.
    sh_toolchain = ctx.toolchains[_SH_TOOLCHAIN_TYPE]
    if not sh_toolchain or not sh_toolchain.path:
        fail("""No suitable shell toolchain found:
* if you are running Bazel on Windows, set the BAZEL_SH environment variable to the path of bash.exe
* if you are running Bazel on a non-Windows platform but are targeting Windows, register an sh_toolchain for the {} toolchain type
""".format(_SH_TOOLCHAIN_TYPE))

    return _create_windows_exe_launcher(ctx, sh_toolchain, primary_output)

def _make_sh_executable_rule(extra_attrs = {}, **kwargs):
    return rule(
        _sh_executable_impl,
        doc = """
<p>
  The <code>sh_binary</code> rule is used to declare executable shell scripts.
  (<code>sh_binary</code> is a misnomer: its outputs aren't necessarily binaries.) This rule ensures
  that all dependencies are built, and appear in the <code>runfiles</code> area at execution time.
  We recommend that you name your <code>sh_binary()</code> rules after the name of the script minus
  the extension (e.g. <code>.sh</code>); the rule name and the file name must be distinct.
  <code>sh_binary</code> respects shebangs, so any available interpreter may be used (eg.
  <code>#!/bin/zsh</code>)
</p>

<h4 id="sh_binary_examples">Example</h4>

<p>For a simple shell script with no dependencies and some data files:
</p>

<pre class="code">
sh_binary(
    name = "foo",
    srcs = ["foo.sh"],
    data = glob(["datafiles/*.txt"]),
)
</pre>
""",
        attrs = {
            "srcs": attr.label_list(
                allow_files = True,
                doc = """
The list of input files.
<p>
  This attribute should be used to list shell script source files that belong to
  this library. Scripts can load other scripts using the shell's <code>source</code>
  or <code>.</code> command.
</p>
""",
            ),
            "data": attr.label_list(
                allow_files = True,
                flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
            ),
            "deps": attr.label_list(
                allow_rules = ["sh_library"],
                doc = """
The list of "library" targets to be aggregated into this target.
See general comments about <code>deps</code>
at <a href="${link common-definitions#typical.deps}">Typical attributes defined by
most build rules</a>.
<p>
  This attribute should be used to list other <code>sh_library</code> rules that provide
  interpreted program source code depended on by the code in <code>srcs</code>. The files
  provided by these rules will be present among the <code>runfiles</code> of this target.
</p>
""",
            ),
            "env": attr.string_dict(),
            "env_inherit": attr.string_list(),
            "_windows_constraint": attr.label(
                default = "@platforms//os:windows",
            ),
        } | extra_attrs,
        toolchains = [
            config_common.toolchain_type(_SH_TOOLCHAIN_TYPE, mandatory = False),
        ],
        **kwargs
    )

sh_binary = _make_sh_executable_rule(executable = True)
sh_test = _make_sh_executable_rule(
    test = True,
    fragments = ["coverage"],
    extra_attrs = {
        "_lcov_merger": attr.label(
            cfg = "exec",
            default = configuration_field(fragment = "coverage", name = "output_generator"),
            executable = True,
        ),
        # Add the script as an attribute in order for sh_test to output code coverage results for
        # code covered by CC binaries invocations.
        "_collect_cc_coverage": attr.label(
            cfg = "exec",
            default = "@bazel_tools//tools/test:collect_cc_coverage",
            executable = True,
        ),
    },
)
