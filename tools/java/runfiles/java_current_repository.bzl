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

_CurrentRepositoryInfo = provider(
    fields = {
        "value": "The canonical name of the repository.",
    },
)

def _current_repository_impl(ctx):
    return _CurrentRepositoryInfo(value = ctx.build_setting_value)

current_repository_setting = rule(
    implementation = _current_repository_impl,
    build_setting = config.string(),
)

def _java_current_repository_impl(ctx):
    java_file = ctx.actions.declare_file("_java_current_repository/RunfilesConstants.java")
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = java_file,
        substitutions = {
            "__CURRENT_REPOSITORY__": ctx.attr._current_repository[_CurrentRepositoryInfo].value,
        },
    )

    # Use javac directly rather than java_common.compile to avoid a Java toolchain dependency.
    # Since this rule is built in a unique configuration per repository name, the bootclasspath
    # rule would otherwise rerun once for every repository using Java.
    class_dir = ctx.actions.declare_directory("_java_current_repository/runfiles_constants")

    javac_args = ctx.actions.args()
    javac_args.add("-source")
    javac_args.add("8")
    javac_args.add("-target")
    javac_args.add("8")
    javac_args.add("-Xlint:-options")
    javac_args.add("-d")
    javac_args.add_all([class_dir], expand_directories = False)
    javac_args.add(java_file)

    java_runtime = ctx.toolchains["@bazel_tools//tools/jdk:runtime_toolchain_type"].java_runtime

    ctx.actions.run(
        executable = "%s/bin/javac" % java_runtime.java_home,
        mnemonic = "JavaCurrentRepositoryCompile",
        inputs = depset([java_file], transitive = [java_runtime.files]),
        outputs = [class_dir],
        arguments = [javac_args],
    )

    jar_file = ctx.actions.declare_file("_java_current_repository/runfiles_constants.jar")

    zipper_args = ctx.actions.args()
    zipper_args.add("c")
    zipper_args.add(jar_file)

    # Add the single .class file at the expected location.
    zipper_args.add_all([class_dir], format_each = "com/google/devtools/build/runfiles/RunfilesConstants.class=%s")

    ctx.actions.run(
        executable = ctx.executable._zipper,
        mnemonic = "JavaCurrentRepositoryJar",
        inputs = [class_dir],
        outputs = [jar_file],
        arguments = [zipper_args],
    )

    # The JLS guarantees that constants are inlined. Since the generated code only contains
    # constants, we can remove it from the runtime classpath. This also serves to prevent
    # confusion as different repositories will compile against different versions of the
    # generated class, with an essentially arbitrary one appearing on the runtime classpath.
    java_info = JavaInfo(
        output_jar = jar_file,
        compile_jar = jar_file,
        neverlink = True,
    )

    return [
        DefaultInfo(files = depset([jar_file])),
        java_info,
    ]

java_current_repository = rule(
    attrs = {
        "_current_repository": attr.label(
            default = ":current_repository",
        ),
        "_template": attr.label(
            default = "RunfilesConstants.java.tpl",
            allow_single_file = True,
        ),
        "_zipper": attr.label(
            default = "@bazel_tools//tools/zip:zipper",
            executable = True,
            cfg = "exec",
        ),
    },
    implementation = _java_current_repository_impl,
    provides = [JavaInfo],
    fragments = ["java"],
    toolchains = [config_common.toolchain_type("@bazel_tools//tools/jdk:runtime_toolchain_type", mandatory = True)],
)
