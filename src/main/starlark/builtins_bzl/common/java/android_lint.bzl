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

load(":common/java/java_semantics.bzl", "semantics")

def android_lint_action(ctx, source_files, source_jars, compilation_info):
    """
    Creates an action that runs Android lint against Java source files.

    You need to add `ANDROID_LINT_IMPLICIT_ATTRS` to any rule or aspect using this call.

    To lint generated source jars (java_info.java_outputs.gen_source_jar)
    add them to the `source_jar` parameter.

    `compilation_info` parameter should supply the classpath and Javac options
    that were used during Java compilation.

    The Android lint tool is obtained from Java toolchain.

    Args:
      ctx: (RuleContext) Used to register the action.
      source_files: (list[File]) A list of .java source files
      source_jars: (list[File])  A list of .jar or .srcjar files containing
        source files. It should also include generated source jars.
      compilation_info: (struct) Information about compilation.

    Returns:
      (None|File) The Android lint output file or None if no source files were
      present.
    """

    # assuming that linting is enabled for all java rules i.e.
    # --experimental_limit_android_lint_to_android_constrained_java=false

    # --experimental_run_android_lint_on_java_rules= is checked in basic_java_library.bzl

    if not (source_files or source_jars):
        return None

    toolchain = semantics.find_java_toolchain(ctx)
    java_runtime = toolchain.java_runtime
    linter = toolchain.android_linter()
    if not linter:
        # TODO(hvd): enable after enabling in tests
        # fail("android linter not set in java_toolchain")
        return None

    args = ctx.actions.args()

    executable = linter.tool.executable
    transitive_inputs = []
    if executable.extension != "jar":
        tools = [linter.tool, linter.data]
        args_list = [args]
    else:
        jvm_args = ctx.actions.args()
        jvm_args.add_all(toolchain.jvm_opt)
        jvm_args.add_all(linter.jvm_opts)
        jvm_args.add("-jar", executable)
        executable = java_runtime.java_executable_exec_path
        tools = [java_runtime.files, linter.tool.executable, linter.data]
        args_list = [jvm_args, args]

    classpath = compilation_info.compilation_classpath

    # TODO(hvd): get from toolchain if we need this - probably android only
    bootclasspath_aux = []
    if bootclasspath_aux:
        classpath = depset(transitive = [classpath, bootclasspath_aux])
    transitive_inputs.append(classpath)

    bootclasspath = toolchain.bootclasspath
    transitive_inputs.append(bootclasspath)

    transitive_inputs.append(compilation_info.plugins.processor_jars)
    transitive_inputs.append(compilation_info.plugins.processor_data)
    args.add_all("--sources", source_files)
    args.add_all("--source_jars", source_jars)
    args.add_all("--bootclasspath", bootclasspath)
    args.add_all("--classpath", classpath)
    args.add_all("--lint_rules", compilation_info.plugins.processor_jars)
    args.add("--target_label", ctx.label)

    javac_opts = compilation_info.javac_options
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
        progress_message = semantics.LINT_PROGRESS_MESSAGE,
        executable = executable,
        inputs = depset(
            # TODO(b/213551463) benchmark using a transitive depset instead
            source_files + source_jars,
            transitive = transitive_inputs,
        ),
        outputs = [android_lint_out],
        tools = tools,
        arguments = args_list,
        execution_requirements = {"supports-workers": "1"},
        toolchain = semantics.JAVA_TOOLCHAIN_TYPE,
        env = {
            # TODO(b/279025786): replace with setting --XskipJarVerification in AndroidLintRunner
            "ANDROID_LINT_SKIP_BYTECODE_VERIFIER": "true",
        },
    )
    return android_lint_out
