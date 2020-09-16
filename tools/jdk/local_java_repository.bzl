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

"""Java repository implementation.

Creates a local repository using jdk.BUILD file.

When Java executable is not present it creates a BUILD file with target "jdk"
displaying an error message.
"""

def _local_java_repository_impl(repository_ctx):
    java_home = repository_ctx.attr.java_home
    java_home_path = repository_ctx.path(java_home)
    if not java_home_path.exists:
        fail('The path indicated by the "java_home" attribute "%s" (absolute: "%s") ' +
             "does not exist." % (java_home, str(java_home_path)))

    repository_ctx.file(
        "WORKSPACE",
        "# DO NOT EDIT: automatically generated WORKSPACE file for local_java_repository\n" +
        "workspace(name = \"{name}\")\n".format(name = repository_ctx.name),
    )

    extension = ".exe" if repository_ctx.os.name.lower().find("windows") != -1 else ""
    if java_home_path.get_child("bin").get_child("java" + extension).exists:
        repository_ctx.file(
            "BUILD.bazel",
            repository_ctx.read(repository_ctx.path(repository_ctx.attr._build_file)),
            False,
        )

        # Symlink all files
        for file in repository_ctx.path(java_home).readdir():
            repository_ctx.symlink(file, file.basename)

        return

    # Java binary does not exist
    # TODO(ilist): replace error message after toolchain implementation
    repository_ctx.file(
        "BUILD.bazel",
        '''load("@bazel_tools//tools/jdk:fail_rule.bzl", "fail_rule")

fail_rule(
    name = "jdk",
    header = "Auto-Configuration Error:",
    message = ("Cannot find Java binary %s in %s; either correct your JAVA_HOME, " +
           "PATH or specify embedded Java (e.g. " +
           "--javabase=@bazel_tools//tools/jdk:remote_jdk11)")
)''' % ("bin/java" + extension, java_home),
        False,
    )

local_java_repository = repository_rule(
    implementation = _local_java_repository_impl,
    local = True,
    configure = True,
    attrs = {
        "java_home": attr.string(),
        "_build_file": attr.label(default = "@bazel_tools//tools/jdk:jdk.BUILD"),
    },
)
