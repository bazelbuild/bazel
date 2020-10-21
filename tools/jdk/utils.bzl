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

def java_runtime_files(name, srcs):
    """Copies the given sources out of the current Java runtime."""

    native.filegroup(
        name = name,
        srcs = srcs,
    )
    for src in srcs:
        native.genrule(
            name = "gen_%s" % src,
            srcs = ["@bazel_tools//tools/jdk:current_java_runtime"],
            toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
            cmd = "cp $(JAVABASE)/%s $@" % src,
            outs = [src],
        )

def _bootclasspath_impl(ctx):
    host_javabase = ctx.attr.host_javabase[java_common.JavaRuntimeInfo]

    # explicitly list output files instead of using TreeArtifact to work around
    # https://github.com/bazelbuild/bazel/issues/6203
    classes = [
        "DumpPlatformClassPath.class",
    ]

    class_outputs = [
        ctx.actions.declare_file("%s_classes/%s" % (ctx.label.name, clazz))
        for clazz in classes
    ]

    args = ctx.actions.args()
    args.add("-source")
    args.add("8")
    args.add("-target")
    args.add("8")
    args.add("-Xlint:-options")
    args.add("-cp")
    args.add("%s/lib/tools.jar" % host_javabase.java_home)
    args.add("-d")
    args.add(class_outputs[0].dirname)
    args.add(ctx.file.src)

    ctx.actions.run(
        executable = "%s/bin/javac" % host_javabase.java_home,
        inputs = [ctx.file.src] + ctx.files.host_javabase,
        outputs = class_outputs,
        arguments = [args],
    )

    bootclasspath = ctx.outputs.output_jar

    inputs = class_outputs + ctx.files.host_javabase

    args = ctx.actions.args()
    args.add("-XX:+IgnoreUnrecognizedVMOptions")
    args.add("--add-exports=jdk.compiler/com.sun.tools.javac.platform=ALL-UNNAMED")
    args.add_joined(
        "-cp",
        [class_outputs[0].dirname, "%s/lib/tools.jar" % host_javabase.java_home],
        join_with = ctx.configuration.host_path_separator,
    )
    args.add("DumpPlatformClassPath")
    args.add(bootclasspath)

    if ctx.attr.target_javabase:
        inputs.extend(ctx.files.target_javabase)
        args.add(ctx.attr.target_javabase[java_common.JavaRuntimeInfo].java_home)

    ctx.actions.run(
        executable = str(host_javabase.java_executable_exec_path),
        inputs = inputs,
        outputs = [bootclasspath],
        arguments = [args],
    )
    return [
        DefaultInfo(files = depset([bootclasspath])),
        OutputGroupInfo(jar = [bootclasspath]),
    ]

_bootclasspath = rule(
    implementation = _bootclasspath_impl,
    attrs = {
        "host_javabase": attr.label(
            cfg = "host",
            providers = [java_common.JavaRuntimeInfo],
        ),
        "src": attr.label(
            cfg = "host",
            allow_single_file = True,
        ),
        "target_javabase": attr.label(
            providers = [java_common.JavaRuntimeInfo],
        ),
        "output_jar": attr.output(mandatory = True),
    },
)

def bootclasspath(name, **kwargs):
    _bootclasspath(
        name = name,
        output_jar = name + ".jar",
        **kwargs
    )
