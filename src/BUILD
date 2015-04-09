# Packaging

genrule(
    name = "client-info-file",
    outs = ["client_info"],
    cmd = "touch $@",
    executable = 1,
)

md5_cmd = "set -e -o pipefail && cat $(SRCS) | %s | awk '{ print $$1; }' > $@"

# TODO(bazel-team): find a better way to handle dylib extensions.
filegroup(
    name = "libunix",
    srcs = select({
        ":darwin": ["//src/main/native:libunix.dylib"],
        "//conditions:default": ["//src/main/native:libunix.so"],
    }),
    visibility = ["//src/test/java:__pkg__"],
)

genrule(
    name = "install_base_key-file",
    srcs = [
        "//src/main/java:bazel-main_deploy.jar",
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        ":namespace-sandbox",
        "client_info",
        "//src/main/tools:build_interface_so",
    ],
    outs = ["install_base_key"],
    cmd = select({
        ":darwin": md5_cmd % "/sbin/md5",
        "//conditions:default": md5_cmd % "md5sum",
    }),
)

genrule(
    name = "java-version",
    outs = ["java.version"],
    cmd = "echo 1.8 >$@",
)

genrule(
    name = "package-zip",
    srcs = [
        "//src/main/java:bazel-main_deploy.jar",
        # The jar must the first in the zip file because the client launcher
        # looks for the first entry in the zip file for the java server.
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:jdk-support",
        ":namespace-sandbox",
        "client_info",
        "//src/main/tools:build_interface_so",
        "install_base_key",
        ":java-version",
    ],
    outs = ["package.zip"],
    # Terrible hack to remove timestamps in the zip file
    cmd = "mkdir $(@D)/package-zip && " +
          "cp $(SRCS) $(@D)/package-zip && " +
          "touch -t 198001010000.00 $(@D)/package-zip/* && " +
          "zip -qj $@ $(@D)/package-zip/* && " +
          "rm -fr $(@D)/package-zip",
)

genrule(
    name = "bazel-bin",
    srcs = [
        "//src/main/cpp:client",
        "package-zip",
    ],
    outs = ["bazel"],
    cmd = "cat $(location //src/main/cpp:client) package-zip > $@ && zip -qA $@",
    executable = 1,
    output_to_bindir = 1,
    visibility = [
        "//scripts:__pkg__",  # For bash completion generation
        "//src/test:__subpackages__",  # For integration tests
    ],
)

filegroup(
    name = "tools",
    srcs = [
        "//src/java_tools/buildjar:JavaBuilder_deploy.jar",
        "//src/java_tools/singlejar:SingleJar_deploy.jar",
        "//third_party/ijar",
    ],
)

filegroup(
    name = "namespace-sandbox",
    srcs = [
        "//src/main/tools:namespace-sandbox",
    ],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)
