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

"""Rules for importing and registering a local JDK."""

def _detect_java_version(repository_ctx, java_bin):
    properties_out = repository_ctx.execute([java_bin, "-XshowSettings:properties"]).stderr
    # This returns an indented list of properties separated with newlines:
    # "  java.vendor.url.bug = ... \n"
    # "  java.version = 11.0.8\n"
    # "  java.version.date = 2020-11-05\"

    strip_properties = [property.strip() for property in properties_out.splitlines()]
    version_property = [property for property in strip_properties if property.startswith("java.version = ")]
    if len(version_property) != 1:
        return "unknown"

    version_value = version_property[0][len("java.version = "):]
    (major, minor, rest) = version_value.split(".", 2)

    if major == "1":  # handles versions below 1.8
        return minor
    return major

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
    java_bin = java_home_path.get_child("bin").get_child("java" + extension)
    if java_bin.exists:
        version = repository_ctx.attr.version if repository_ctx.attr.version != "" else _detect_java_version(repository_ctx, java_bin)

        repository_ctx.file(
            "BUILD.bazel",
            repository_ctx.read(repository_ctx.path(repository_ctx.attr._build_file)) +
            """
config_setting(
    name = "name_setting",
    values = {{"java_runtime_version": "{local_jdk}"}},
    visibility = ["//visibility:private"],
)
config_setting(
    name = "version_setting",
    values = {{"java_runtime_version": "{version}"}},
    visibility = ["//visibility:private"],
)
alias(
    name = "version_or_name_setting",
    actual = select({{
        ":version_setting": ":version_setting",
        "//conditions:default": ":name_setting",
    }}),
    visibility = ["//visibility:private"],
)
toolchain(
    name = "toolchain",
    target_settings = [":version_or_name_setting"],
    toolchain_type = "@bazel_tools//tools/jdk:runtime_toolchain_type",
    toolchain = ":jdk",
)
""".format(local_jdk = repository_ctx.name, version = version),
            False,
        )

        # Symlink all files
        for file in repository_ctx.path(java_home).readdir():
            repository_ctx.symlink(file, file.basename)

        return

    # Java binary does not exist
    repository_ctx.file(
        "BUILD.bazel",
        '''load("@bazel_tools//tools/jdk:fail_rule.bzl", "fail_rule")

fail_rule(
    name = "jdk",
    header = "Auto-Configuration Error:",
    message = ("Cannot find Java binary {java_binary} in {java_home}; either correct your JAVA_HOME, " +
           "PATH or specify Java from remote repository (e.g. " +
           "--java_runtime_version=remotejdk_11")
)
config_setting(
    name = "localjdk_setting",
    values = {{"java_runtime_version": "{local_jdk}"}},
    visibility = ["//visibility:private"],
)
toolchain(
    name = "toolchain",
    target_settings = [":localjdk_setting"],
    toolchain_type = "@bazel_tools//tools/jdk:runtime_toolchain_type",
    toolchain = ":jdk",
)
'''.format(
            local_jdk = repository_ctx.name,
            java_binary = "bin/java" + extension,
            java_home = java_home,
        ),
        False,
    )

_local_java_repository_rule = repository_rule(
    implementation = _local_java_repository_impl,
    local = True,
    configure = True,
    attrs = {
        "java_home": attr.string(),
        "version": attr.string(),
        "_build_file": attr.label(default = "@bazel_tools//tools/jdk:jdk.BUILD"),
    },
)

def local_java_repository(name, java_home, version = ""):
    """Imports and registers a local JDK.

    Toolchain resolution is constrained with --java_runtime_version flag
    having value of the "name" parameter.

    Args:
      name: A unique name for this rule.
      java_home: Location of the JDK imported.
      version: optionally java version
    """
    _local_java_repository_rule(name = name, java_home = java_home, version = version)
    native.register_toolchains("@" + name + "//:toolchain")
