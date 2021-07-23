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

"""Creates the android lint action for java rules"""

load(":common/rule_util.bzl", "create_dep")

java_common = _builtins.toplevel.java_common

def _impl(ctx, java_info, source_files, source_jars):
    # assuming that linting is enabled for all java rules i.e.
    # --experimental_run_android_lint_on_java_rules=true and
    # --experimental_limit_android_lint_to_android_constrained_java=false
    if (ctx.configuration == ctx.host_configuration or
        ctx.bin_dir.path.find("-exec-") >= 0):
        return None

    srcs = ctx.files.srcs
    if not srcs or (hasattr(ctx.attr, "neverlink") and ctx.attr.neverlink):
        return None

    toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo]
    java_runtime = toolchain.java_runtime
    linter = toolchain.android_linter()
    if not linter:
        # TODO(hvd): enable after enabling in tests
        # fail("android linter not set in java_toolchain")
        return None

    args = ctx.actions.args()

    tool = linter.tool
    executable = linter.tool.executable
    transitive_inputs = [linter.data]
    if executable.extension == "jar":
        args.add_all(toolchain.jvm_opt)
        args.add("-jar", executable)
        executable = java_runtime.java_executable_exec_path
        transitive_inputs.append(java_runtime.files)

    for output in java_info.java_outputs:
        if output.generated_source_jar != None:
            source_jars.append(output.generated_source_jar)

    # TODO(ilist): collect compile_jars from JavaInfo in deps & exports
    classpath = java_info.compilation_info.compilation_classpath

    # TODO(hvd): get from toolchain if we need this - probably android only
    bootclasspath_aux = []
    if bootclasspath_aux:
        classpath = depset(transitive = [classpath, bootclasspath_aux])
    transitive_inputs.append(classpath)

    bootclasspath = toolchain.bootclasspath
    transitive_inputs.append(bootclasspath)

    plugin_info = java_info.plugins
    transitive_inputs.append(plugin_info.processor_jars)
    transitive_inputs.append(plugin_info.processor_data)

    args.add_all("--sources", source_files)
    args.add_all("--source_jars", source_jars)
    args.add_all("--bootclasspath", bootclasspath)
    args.add_all("--classpath", classpath)
    args.add_all("--plugins", plugin_info.processor_jars)
    args.add("--target_label", ctx.label)

    javac_opts = java_info.compilation_info.javac_options
    if (javac_opts):
        args.add_all("--javacopts", javac_opts)
        args.add("--")

    args.add("--lintopts")
    args.add_all(linter.lint_opts)

    for package_config in linter.package_config:
        if package_config.matches(ctx.label):
            args.add_all(package_config.javac_opts())
            transitive_inputs.append(package_config.data())

    android_lint_out = ctx.actions.declare_file("%s_android_lint_output.xml" % ctx.label.name)
    args.add("--xml", android_lint_out)

    args.set_param_file_format(format = "multiline")
    args.use_param_file(param_file_arg = "@%s", use_always = True)
    ctx.actions.run(
        mnemonic = "AndroidLint",
        progress_message = "Running Android Lint for: %{label}",
        executable = executable,
        inputs = depset(
            source_files + source_jars,
            transitive = transitive_inputs,
        ),
        outputs = [android_lint_out],
        tools = [tool],
        arguments = [args],
        execution_requirements = {"supports-workers": "1"},
    )
    return android_lint_out

ANDROID_LINT_ACTION = create_dep(
    call = _impl,
    attrs = {
        "_java_toolchain": attr.label(
            default = "@//tools/jdk:current_java_toolchain",
            providers = [java_common.JavaToolchainInfo],
        ),
    },
)
