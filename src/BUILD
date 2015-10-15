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
        "//src/test/java:__pkg__",
        "//src/tools/generate_workspace:__pkg__",
    ],
)

[genrule(
    name = "install_base_key-file" + suffix,
    srcs = [
        "//src/main/java:bazel-main_deploy.jar",
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:build_interface_so",
    ] + embedded_tools,
    outs = ["install_base_key" + suffix],
    cmd = select({
        ":darwin": md5_cmd % "/sbin/md5",
        "//conditions:default": md5_cmd % "md5sum",
    }),
) for suffix, embedded_tools in {
    "": [":embedded_tools"],
    "_notools": [],
}.items()]

# Try to grab the java version from the java_toolchain.
# Unfortunately, we don't have access to the javac options
# so we cannot get it from the command-line options.
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
            echo "1.8" >$@  # Java 8 is the default
          elif [[ "$${JAVA_VERSION}" =~ ^[0-9]+$$ ]]; then
            echo "1.$${JAVA_VERSION}" >$@  # Add 1. before 7 or 8
          else
            echo "$${JAVA_VERSION}" >$@
          fi
          """,
)

genrule(
    name = "embedded_tools",
    srcs = [
        "//tools:embedded_tools_srcs",
        "//third_party:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:srcs",
        "//src/tools/android/java/com/google/devtools/build/android:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:srcs",
        "//src/main/protobuf:srcs",
        "//src/main/java:options-srcs",
    ],
    outs = ["embedded_tools.zip"],
    cmd = "\n".join([
        "mkdir -p $(@D)/embedded-tools-zip",
        "for i in $(SRCS); do",
        "  mkdir -p $$(dirname $(@D)/embedded-tools-zip/$$i)",
        "  cp $$i $$(dirname $(@D)/embedded-tools-zip/$$i)",
        "done",
        "touch $(@D)/embedded-tools-zip/WORKSPACE",
        "mkdir -p $(@D)/embedded-tools-zip/tools/defaults",
        "touch $(@D)/embedded-tools-zip/tools/defaults/BUILD",
        "mv $(@D)/embedded-tools-zip/src/main/java/options.BUILD" +
        " $(@D)/embedded-tools-zip/src/main/java/BUILD",
        "find $(@D)/embedded-tools-zip -exec touch -t 198001010000.00 '{}' ';'",
        "P=$$PWD; (cd $(@D)/embedded-tools-zip && zip -qrD $$P/$@ *)",
        "rm -fr $(@D)/embedded-tools-zip",
    ]),
)

[genrule(
    name = "package-zip" + suffix,
    srcs = [
        # The script assumes that the deploy jar is the first item here, the
        # install base key is the second, and the embedded tools zip (if exists)
        # is the third
        "//src/main/java:bazel-main_deploy.jar",
        "install_base_key" + suffix,
    ] + ([":embedded_tools.zip"] if embed else []) + [
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:jdk-support",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:build_interface_so",
        ":java-version",
    ],
    outs = ["package" + suffix + ".zip"],
    cmd = "$(location :package-bazel.sh) $@ " + ("" if embed else "''") + " $(SRCS)",
    tools = ["package-bazel.sh"],
) for suffix, embed in [
    ("", True),
    ("_notools", False),
]]

[genrule(
    name = "bazel-bin" + suffix,
    srcs = [
        "//src/main/cpp:client",
        "package-zip" + suffix,
    ],
    outs = ["bazel" + suffix],
    cmd = "cat $(location //src/main/cpp:client) $(location :package-zip" + suffix + ") > $@ && zip -qA $@",
    executable = 1,
    output_to_bindir = 1,
    visibility = [
        "//scripts:__pkg__",  # For bash completion generation
        "//scripts/packages:__pkg__",  # For installer generation
        "//src/test:__subpackages__",  # For integration tests
    ],
) for suffix in [
    "",
    "_notools",
]]

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)
