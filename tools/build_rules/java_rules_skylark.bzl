# Copyright 2014 The Bazel Authors. All rights reserved.
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

"""Java rules for bootstraping Bazel.


This is a quick and dirty rule to make Bazel compile itself. It's not production
ready.
"""

_JarsInfo = provider(fields = ["compile_time_jars", "runtime_jars"])

def _java_library_impl(ctx):
    javac_options = ctx.fragments.java.default_javac_flags
    class_jar = ctx.outputs.class_jar
    compile_time_jars = depset(order = "topological")
    runtime_jars = depset(order = "topological")
    for dep in ctx.attr.deps:
        compile_time_jars = depset(
            transitive = [compile_time_jars, dep[_JarsInfo].compile_time_jars],
        )
        runtime_jars = depset(
            transitive = [runtime_jars, dep[_JarsInfo].runtime_jars],
        )

    jars = ctx.files.jars
    neverlink_jars = ctx.files.neverlink_jars
    compile_time_jars = depset(jars + neverlink_jars, transitive = [compile_time_jars])
    runtime_jars = depset(jars, transitive = [runtime_jars])
    compile_time_jars_list = compile_time_jars.to_list()  # TODO: This is weird.

    build_output = class_jar.path + ".build_output"
    java_output = class_jar.path + ".build_java"
    javalist_output = class_jar.path + ".build_java_list"
    sources = ctx.files.srcs

    sources_param_file = ctx.actions.declare_file(class_jar.basename + "-2.params")
    ctx.actions.write(
        output = sources_param_file,
        content = cmd_helper.join_paths("\n", depset(sources)),
        is_executable = False,
    )

    # Cleaning build output directory
    cmd = "set -e;rm -rf " + build_output + " " + java_output + " " + javalist_output + "\n"
    cmd += "mkdir " + build_output + " " + java_output + "\n"
    files = " @" + sources_param_file.path
    java_runtime = ctx.attr._jdk[java_common.JavaRuntimeInfo]
    jar_path = "%s/bin/jar" % java_runtime.java_home

    if ctx.files.srcjars:
        files += " @" + javalist_output
        for file in ctx.files.srcjars:
            cmd += "%s tf %s | grep '\\.java$' | sed 's|^|%s/|' >> %s\n" % (jar_path, file.path, java_output, javalist_output)
            cmd += "unzip %s -d %s >/dev/null\n" % (file.path, java_output)

    if ctx.files.srcs or ctx.files.srcjars:
        cmd += "%s/bin/javac" % java_runtime.java_home
        cmd += " " + " ".join(javac_options)
        if compile_time_jars:
            cmd += " -classpath '" + cmd_helper.join_paths(ctx.configuration.host_path_separator, compile_time_jars) + "'"
        cmd += " -d " + build_output + files + "\n"

    # We haven't got a good story for where these should end up, so
    # stick them in the root of the jar.
    for r in ctx.files.resources:
        cmd += "cp %s %s\n" % (r.path, build_output)
    cmd += (jar_path + " cf " + class_jar.path + " -C " + build_output + " .\n" +
            "touch " + build_output + "\n")
    ctx.actions.run_shell(
        inputs = (sources + compile_time_jars_list + [sources_param_file] +
                  ctx.files._jdk + ctx.files.resources + ctx.files.srcjars),
        outputs = [class_jar],
        mnemonic = "JavacBootstrap",
        command = cmd,
        use_default_shell_env = True,
    )

    runfiles = ctx.runfiles(collect_data = True)

    compile_time_jars = depset(transitive = [compile_time_jars], direct = [class_jar])
    runtime_jars = depset(transitive = [runtime_jars], direct = [class_jar])

    return [
        DefaultInfo(
            files = depset([class_jar]),
            runfiles = runfiles,
        ),
        _JarsInfo(
            compile_time_jars = compile_time_jars,
            runtime_jars = runtime_jars,
        ),
    ]

def _java_binary_impl(ctx):
    library_result = _java_library_impl(ctx)

    deploy_jar = ctx.outputs.deploy_jar
    manifest = ctx.outputs.manifest
    build_output = deploy_jar.path + ".build_output"
    main_class = ctx.attr.main_class
    java_runtime = ctx.attr._jdk[java_common.JavaRuntimeInfo]
    jar_path = "%s/bin/jar" % java_runtime.java_home
    ctx.actions.write(
        output = manifest,
        content = "Main-Class: " + main_class + "\n",
        is_executable = False,
    )

    # Cleaning build output directory
    cmd = "set -e;rm -rf " + build_output + ";mkdir " + build_output + "\n"
    for jar in library_result[1].runtime_jars.to_list():
        cmd += "unzip -qn " + jar.path + " -d " + build_output + "\n"
    cmd += (jar_path + " cmf " + manifest.path + " " +
            deploy_jar.path + " -C " + build_output + " .\n" +
            "touch " + build_output + "\n")

    ctx.actions.run_shell(
        inputs = library_result[1].runtime_jars.to_list() + [manifest] + ctx.files._jdk,
        outputs = [deploy_jar],
        mnemonic = "Deployjar",
        command = cmd,
        use_default_shell_env = True,
    )

    # Write the wrapper.
    executable = ctx.outputs.executable
    ctx.actions.write(
        output = executable,
        content = "\n".join([
            "#!/bin/bash",
            "# autogenerated - do not edit.",
            "case \"$0\" in",
            "/*) self=\"$0\" ;;",
            "*)  self=\"$PWD/$0\";;",
            "esac",
            "",
            "if [[ -z \"$JAVA_RUNFILES\" ]]; then",
            "  if [[ -e \"${self}.runfiles\" ]]; then",
            "    export JAVA_RUNFILES=\"${self}.runfiles\"",
            "  fi",
            "  if [[ -n \"$JAVA_RUNFILES\" ]]; then",
            "    export TEST_SRCDIR=${TEST_SRCDIR:-$JAVA_RUNFILES}",
            "  fi",
            "fi",
            "",
            "jvm_bin=%s" % (ctx.attr._jdk[java_common.JavaRuntimeInfo].java_executable_exec_path),
            "if [[ ! -x ${jvm_bin} ]]; then",
            "  jvm_bin=$(which java)",
            "fi",

            # We extract the .so into a temp dir. If only we could mmap
            # directly from the zip file.
            "DEPLOY=$(dirname $self)/$(basename %s)" % deploy_jar.path,

            # This works both on Darwin and Linux, with the darwin path
            # looking like tmp.XXXXXXXX.{random}
            "SO_DIR=$(mktemp -d -t tmp.XXXXXXXXX)",
            "function cleanup() {",
            "  rm -rf ${SO_DIR}",
            "}",
            "trap cleanup EXIT",
            "unzip -q -d ${SO_DIR} ${DEPLOY} \"*.so\" \"*.dll\" \"*.dylib\" >& /dev/null",
            ("${jvm_bin} -Djava.library.path=${SO_DIR} %s -jar $DEPLOY \"$@\"" %
             " ".join(ctx.attr.jvm_flags)),
            "",
        ]),
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = [deploy_jar, executable] + ctx.files._jdk, collect_data = True)
    files_to_build = depset(
        transitive = [library_result[0].files],
        direct = [deploy_jar, manifest, executable],
    )

    return [DefaultInfo(files = files_to_build, runfiles = runfiles)]

def _java_import_impl(ctx):
    # TODO(bazel-team): Why do we need to filter here? The attribute
    # already says only jars are allowed.
    jars = depset(ctx.files.jars)
    neverlink_jars = depset(ctx.files.neverlink_jars)
    runfiles = ctx.runfiles(collect_data = True)
    compile_time_jars = depset(transitive = [jars, neverlink_jars])

    return [
        DefaultInfo(files = jars, runfiles = runfiles),
        _JarsInfo(
            compile_time_jars = compile_time_jars,
            runtime_jars = jars,
        ),
    ]

java_library_attrs = {
    "_jdk": attr.label(
        default = Label("//tools/jdk:current_java_runtime"),
        providers = [java_common.JavaRuntimeInfo],
    ),
    "data": attr.label_list(allow_files = True),
    "resources": attr.label_list(allow_files = True),
    "srcs": attr.label_list(allow_files = [".java"]),
    "jars": attr.label_list(allow_files = [".jar"]),
    "neverlink_jars": attr.label_list(allow_files = [".jar"]),
    "srcjars": attr.label_list(allow_files = [".jar", ".srcjar"]),
    "deps": attr.label_list(
        allow_files = False,
        providers = [_JarsInfo],
    ),
}

java_library = rule(
    _java_library_impl,
    attrs = java_library_attrs,
    outputs = {
        "class_jar": "lib%{name}.jar",
    },
    fragments = ["java", "cpp"],
)

# A copy to avoid conflict with native rule.
bootstrap_java_library = rule(
    _java_library_impl,
    attrs = java_library_attrs,
    outputs = {
        "class_jar": "lib%{name}.jar",
    },
    fragments = ["java"],
)

java_binary_attrs_common = dict(java_library_attrs)
java_binary_attrs_common.update({
    "jvm_flags": attr.string_list(),
    "jvm": attr.label(default = Label("//tools/jdk:jdk"), allow_files = True),
})

java_binary_attrs = dict(java_binary_attrs_common)
java_binary_attrs["main_class"] = attr.string(mandatory = True)

java_binary_outputs = {
    "class_jar": "lib%{name}.jar",
    "deploy_jar": "%{name}_deploy.jar",
    "manifest": "%{name}_MANIFEST.MF",
}

java_binary = rule(
    _java_binary_impl,
    executable = True,
    attrs = java_binary_attrs,
    outputs = java_binary_outputs,
    fragments = ["java", "cpp"],
)

# A copy to avoid conflict with native rule
bootstrap_java_binary = rule(
    _java_binary_impl,
    executable = True,
    attrs = java_binary_attrs,
    outputs = java_binary_outputs,
    fragments = ["java"],
)

java_test = rule(
    _java_binary_impl,
    executable = True,
    attrs = dict(list(java_binary_attrs_common.items()) + [
        ("main_class", attr.string(default = "org.junit.runner.JUnitCore")),
        # TODO(bazel-team): it would be better if we could offer a
        # test_class attribute, but the "args" attribute is hard
        # coded in the bazel infrastructure.
    ]),
    outputs = java_binary_outputs,
    test = True,
    fragments = ["java", "cpp"],
)

java_import = rule(
    _java_import_impl,
    attrs = {
        "jars": attr.label_list(allow_files = [".jar"]),
        "srcjar": attr.label(allow_files = [".jar", ".srcjar"]),
        "neverlink_jars": attr.label_list(allow_files = [".jar"], default = []),
    },
)
