# Packaging

md5_cmd = "set -e -o pipefail && cat $(SRCS) | %s | awk '{ print $$1; }' > $@"

# TODO(bazel-team): find a better way to handle dylib extensions.
filegroup(
    name = "libunix",
    srcs = select({
        ":darwin": ["//src/main/native:libunix.dylib"],
        "//conditions:default": ["//src/main/native:libunix.so"],
    }),
    visibility = [
        "//src/main/java/com/google/devtools/build/workspace:__pkg__",
        "//src/test/java:__pkg__",
    ],
)

genrule(
    name = "install_base_key-file",
    srcs = [
        "//src/main/java:bazel-main_deploy.jar",
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:build_interface_so",
    ],
    outs = ["install_base_key"],
    cmd = select({
        ":darwin": md5_cmd % "/sbin/md5",
        "//conditions:default": md5_cmd % "md5sum",
    }),
)

# Try to grab the java version from the java_toolchain.
# Unfortunately, we don't the javac options so we cannot get it from
# the command-line options.
genquery(
    name = "java_toolchain_content",
    expression = "kind(java_toolchain, deps(//tools/defaults:java_toolchain))",
    opts = ["--output=xml"],
    scope = ["//tools/defaults:java_toolchain"],
)

genrule(
    name = "java-version",
    srcs = [":java_toolchain_content"],
    outs = ["java.version"],
    cmd = """
          VERSION_LINE=$$(cat $< | grep target_version);
          JAVA_VERSION=$$(echo $${VERSION_LINE} | sed -E 's/.*value="([^"])".*/\\1/');
          if [ -z "$${JAVA_VERSION}" ]; then
            echo "1.7" >$@  # Java 7 is the default
          elif [[ "$${JAVA_VERSION}" =~ ^[0-9]+$$ ]]; then
            echo "1.$${JAVA_VERSION}" >$@  # Add 1. before 7 or 8
          else
            echo "$${JAVA_VERSION}" >$@
          fi
          """,
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
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:build_interface_so",
        "install_base_key",
        ":java-version",
    ],
    outs = ["package.zip"],
    # Terrible hack to remove timestamps in the zip file
    cmd = "mkdir -p $(@D)/package-zip && " +
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
    cmd = "cat $(location //src/main/cpp:client) $(location :package-zip) > $@ && zip -qA $@",
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

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)
