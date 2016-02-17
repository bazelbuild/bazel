# Packaging

md5_cmd = "set -e -o pipefail && cat $(SRCS) | sort | %s | awk '{ print $$1; }' > $@"

# TODO(bazel-team): find a better way to handle dylib extensions.
filegroup(
    name = "libunix",
    srcs = select({
        ":darwin": ["//src/main/native:libunix.dylib"],
        ":darwin_x86_64": ["//src/main/native:libunix.dylib"],
        ":windows": [],
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
        "//src/main/java/com/google/devtools/build/lib:bazel/BazelServer_deploy.jar",
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:xcode-locator",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:build_interface_so",
    ] + embedded_tools,
    outs = ["install_base_key" + suffix],
    cmd = select({
        ":darwin": md5_cmd % "/sbin/md5",
        ":darwin_x86_64": md5_cmd % "/sbin/md5",
        ":freebsd": md5_cmd % "/sbin/md5",
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
          JAVA_VERSION=$$(echo $${VERSION_LINE} | sed 's/.*value="\\([^"]\\)".*/\\1/');
          if [ -z "$${JAVA_VERSION}" ]; then
            echo "1.8" >$@  # Java 8 is the default
          elif [[ "$${JAVA_VERSION}" =~ ^[0-9]+$$ ]]; then
            echo "1.$${JAVA_VERSION}" >$@  # Add 1. before 7 or 8
          else
            echo "$${JAVA_VERSION}" >$@
          fi
          """,
)

filegroup(
    name = "darwin_tools",
    srcs = [
        "//src/tools/xcode/actoolwrapper:actoolwrapper",
        "//src/tools/xcode/ibtoolwrapper:ibtoolwrapper",
        "//src/tools/xcode/momcwrapper:momcwrapper",
        "//src/tools/xcode/swiftstdlibtoolwrapper:swiftstdlibtoolwrapper",
        "//src/tools/xcode/environment:environment_plist",
        "//src/tools/xcode/xcrunwrapper:xcrunwrapper",
        "//src/objc_tools/bundlemerge:bundlemerge_deploy.jar",
        "//src/objc_tools/plmerge:plmerge_deploy.jar",
        "//src/objc_tools/xcodegen:xcodegen_deploy.jar",
        "//src/tools/xcode/realpath:realpath",
    ] + select({
        ":iphonesdk": ["//src/tools/xcode/stdredirect:StdRedirect.dylib"],
        "//conditions:default": [],
    }),
)

genrule(
    name = "embedded_tools",
    srcs = [
        ":create_embedded_tools.sh",
        "//tools:embedded_tools_srcs",
        "//third_party:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:srcs",
        "//src/tools/android/java/com/google/devtools/build/android:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:embedded_tools",
        "//src/main/protobuf:srcs",
        "//src/java_tools/buildjar:JavaBuilder_deploy.jar",
        "//src/java_tools/singlejar:SingleJar_deploy.jar",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar",
        "//third_party/ijar",
        "//third_party/java/apkbuilder:embedded_tools",
    ] + select({
        ":darwin": [
            ":darwin_tools",
            "//third_party/ijar:zipper",
        ],
        ":darwin_x86_64": [
            ":darwin_tools",
            "//third_party/ijar:zipper",
        ],
        ":windows": [],  # Windows specifically exclude zipper for now.
        "//conditions:default": ["//third_party/ijar:zipper"],
    }),
    outs = ["embedded_tools.zip"],
    cmd = "$(location :create_embedded_tools.sh) $@ $(SRCS)",
)

[genrule(
    name = "package-zip" + suffix,
    srcs = ([":embedded_tools.zip"] if embed else []) + [
        # The script assumes that the embedded tools zip (if exists) is the
        # first item here, the deploy jar the second and install base key is the
        # third
        "//src/main/java/com/google/devtools/build/lib:bazel/BazelServer_deploy.jar",
        "install_base_key" + suffix,
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:xcode-locator",
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

# Build an executable named `bazel.exe`.
# Works on all platforms but only makes sense on Windows.
# Workaround for non-configurability of genrule's `outs` attribute.
[genrule(
    name = "bazel-bin" + suffix + ".exe",
    srcs = [":bazel-bin" + suffix],
    outs = ["bazel" + suffix + ".exe"],
    cmd = "cp $(location :bazel-bin" + suffix + ") $@",
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
    name = "darwin_x86_64",
    values = {"cpu": "darwin_x86_64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "iphonesdk",
    values = {"define": "IPHONE_SDK=1"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)
