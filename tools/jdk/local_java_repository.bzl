"""Java repository implementation

Creates a local repository using jdk.BUILD file.

When java executable is not present it creates a BUILD file with target "jdk"
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
        '''load("@bazel_tools//tools/jdk:local_java_repository.bzl", "failrule")

failrule(
    name = "jdk",
    msg = ("Cannot find Java binary %s in %s; either correct your JAVA_HOME, " +
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

def _fail_rule_impl(ctx):
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, ctx.attr.msg))

failrule = rule(
    implementation = _fail_rule_impl,
    attrs = {"msg": attr.string()},
)
