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
        ":android_tools_repository_zip",
    ],
    outs = ["install_base_key"],
    cmd = select({
        ":darwin": md5_cmd % "/sbin/md5",
        "//conditions:default": md5_cmd % "md5sum",
    }),
)

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
    name = "android_tools_repository_zip",
    srcs = [
        "//tools:srcs",
        "//third_party:srcs",
        "//src/tools/android/java/com/google/devtools/build/android:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:srcs",
        "//src/main/protobuf:srcs",
        "//src/main/java:options-srcs",
    ],
    outs = ["android_tools_repository.zip"],
    cmd = "\n".join([
        "mkdir -p $(@D)/android-tools-zip",
        "for i in $(SRCS); do",
        "  mkdir -p $$(dirname $(@D)/android-tools-zip/$$i)",
        "  cp $$i $$(dirname $(@D)/android-tools-zip/$$i)",
        "done",
        "touch $(@D)/android-tools-zip/WORKSPACE",
        "mv $(@D)/android-tools-zip/src/main/java/options.BUILD" +
        " $(@D)/android-tools-zip/src/main/java/BUILD",
        "find $(@D)/android-tools-zip -exec touch -t 198001010000.00 '{}' ';'",
        "P=$$PWD; (cd $(@D)/android-tools-zip && zip -qrD $$P/$@ *)",
        "rm -fr $(@D)/android-tools-zip",
    ]),
)

genrule(
    name = "package-zip",
    srcs = [
        "//src/main/java:bazel-main_deploy.jar",
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:jdk-support",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:build_interface_so",
        "install_base_key",
        ":java-version",
        ":android_tools_repository_zip",
    ],
    outs = ["package.zip"],
    # Terrible hack to remove timestamps in the zip file
    cmd = "\n".join([
        "mkdir -p $(@D)/package-zip",
        "cp $(SRCS) $(@D)/package-zip",
        # TODO(dmarting): we should change the client to connect to server.jar
        # instead of the first binary in the list.
        "mv $(@D)/package-zip/bazel-main_deploy.jar $(@D)/package-zip/A-server.jar",
        "touch -t 198001010000.00 $(@D)/package-zip/*",
        "mkdir $(@D)/package-zip/android_tools",
        "(cd $(@D)/package-zip/android_tools && unzip -q ../android_tools_repository.zip)",
        "rm $(@D)/package-zip/android_tools_repository.zip",
        "P=$$PWD; (cd $(@D)/package-zip && zip -qrD $$P/$@ *)",
        "rm -fr $(@D)/package-zip",
    ]),
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
        "//scripts/packages:__pkg__",  # For installer generation
        "//src/test:__subpackages__",  # For integration tests
    ],
)

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
