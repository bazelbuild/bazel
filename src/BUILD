# Packaging

md5_cmd = "set -e -o pipefail && cat $(SRCS) | sort | %s | awk '{ print $$1; }' > $@"

# TODO(bazel-team): find a better way to handle dylib extensions.
filegroup(
    name = "libunix",
    srcs = select({
        ":darwin": ["//src/main/native:libunix.dylib"],
        ":darwin_x86_64": ["//src/main/native:libunix.dylib"],
        ":windows": ["//src/main/native:windows_jni.dll"],
        ":windows_msys": ["//src/main/native:windows_jni.dll"],
        ":windows_msvc": ["//src/main/native:windows_jni.dll"],
        "//conditions:default": ["//src/main/native:libunix.so"],
    }),
    visibility = [
        "//src/test/java:__pkg__",
        "//src/tools/generate_workspace:__pkg__",
        "//src/tools/remote_worker:__subpackages__",
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
        "//src/main/tools:linux-sandbox",
        "//tools/osx:xcode-locator",
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
    "_with_jdk": [":embedded_tools_with_jdk"],
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
          JAVA_VERSION=$$(echo $${VERSION_LINE} | sed 's/.*value=\\"\\([^\\"]\\)\\".*/\\1/');
          if [ -z "$${JAVA_VERSION}" ]; then
            echo "1.8" >$@  # Java 8 is the default
          elif [[ "$${JAVA_VERSION}" =~ ^[0-9]+$$ ]]; then
            echo "1.$${JAVA_VERSION}" >$@  # Add 1. before 7 or 8
          else
            echo "$${JAVA_VERSION}" >$@
          fi
          """,
)

# Create dummy tools so we can do select to prevent building iOS target on
# Linux.
OSX_DUMMY_TARGETS = [
    "src/tools/xcode/actoolwrapper/actoolwrapper",
    "src/tools/xcode/ibtoolwrapper/ibtoolwrapper",
    "src/tools/xcode/momcwrapper/momcwrapper",
    "src/tools/xcode/swiftstdlibtoolwrapper/swiftstdlibtoolwrapper",
    "src/tools/xcode/environment/environment_plist",
    "src/objc_tools/bundlemerge/bundlemerge_deploy.jar",
    "src/objc_tools/plmerge/plmerge_deploy.jar",
    "src/objc_tools/xcodegen/xcodegen_deploy.jar",
    "src/tools/xcode/realpath/realpath",
    "src/tools/xcode/stdredirect/StdRedirect.dylib",
]

[
    genrule(
        name = "dummy_darwin_tool_" + name.replace("/", "_"),
        srcs = [],
        outs = ["dummy_darwin_tools/" + name],
        cmd = "touch $@",
    )
    for name in OSX_DUMMY_TARGETS
]

filegroup(
    name = "darwin_tools",
    srcs = [
        "//src/tools/xcode/actoolwrapper:actoolwrapper",
        "//src/tools/xcode/ibtoolwrapper:ibtoolwrapper",
        "//src/tools/xcode/momcwrapper:momcwrapper",
        "//src/tools/xcode/swiftstdlibtoolwrapper:swiftstdlibtoolwrapper",
        "//src/tools/xcode/environment:environment_plist",
        "//src/objc_tools/bundlemerge:bundlemerge_deploy.jar",
        "//src/objc_tools/plmerge:plmerge_deploy.jar",
        "//src/objc_tools/xcodegen:xcodegen_deploy.jar",
        "//src/tools/xcode/realpath:realpath",
        "//tools/osx:xcode_locator.m",
    ] + select({
        ":iphonesdk": ["//src/tools/xcode/stdredirect:StdRedirect.dylib"],
        "//conditions:default": [":dummy_darwin_tool_src_tools_xcode_stdredirect_StdRedirect.dylib"],
    }),
)

filegroup(
    name = "dummy_darwin_tools",
    srcs = [
        "dummy_darwin_tool_" + name.replace("/", "_")
        for name in OSX_DUMMY_TARGETS
    ],
)

[genrule(
    name = "embedded_tools" + suffix,
    srcs = [
        ":create_embedded_tools.sh",
        "//tools:embedded_tools_srcs",
        "//third_party:gpl-srcs",
        "//third_party/java/jarjar:embedded_tools_srcs",
        "//third_party/java/jdk/langtools:test-srcs",
        "//third_party/py/concurrent:srcs",
        "//third_party/py/gflags:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:srcs",
        "//src/tools/android/java/com/google/devtools/build/android:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:embedded_tools",
        "//src/main/protobuf:srcs",
        "//src/java_tools/buildjar:JavaBuilderDeploy",
        "//src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar",
        "//src/java_tools/singlejar:SingleJar_deploy.jar",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:ExperimentalRunner_deploy.jar",
        "//src/java_tools/junitrunner/java/com/google/testing/coverage:embedded_tools",
        "//third_party/ijar",
        "//third_party/java/jacoco:srcs",
    ] + select({
        ":darwin": [
            ":darwin_tools",
            "//third_party/ijar:zipper",
        ],
        ":darwin_x86_64": [
            ":darwin_tools",
            "//third_party/ijar:zipper",
        ],
        "//conditions:default": [
            ":dummy_darwin_tools",
            "//third_party/ijar:zipper",
        ],
    }) + select({
        "//tools/jdk:jdk7": [],
        "//conditions:default": [
            "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_deploy.jar",
        ],
    }) + (select({
        ":darwin": [
            "@openjdk_macos//file",
        ],
        ":darwin_x86_64": [
            "@openjdk_macos//file",
        ],
        ":windows": [
            "@openjdk_win//file",
        ],
        ":windows_msvc": [
            "@openjdk_win//file",
        ],
        "//conditions:default": [
            "@openjdk_linux//file",
        ],
    }) if (suffix == "_with_jdk") else []),
    outs = ["embedded_tools" + suffix + ".zip"],
    cmd = "$(location :create_embedded_tools.sh) $@ $(SRCS)",
) for suffix in [
    "",
    "_with_jdk",
]]

[genrule(
    name = "package-zip" + suffix,
    srcs = ([":embedded_tools" + suffix + ".zip"] if embed else []) + [
        # The script assumes that the embedded tools zip (if exists) is the
        # first item here, the deploy jar the second and install base key is the
        # third
        "//src/main/java/com/google/devtools/build/lib:bazel/BazelServer_deploy.jar",
        "install_base_key" + suffix,
        "//src/main/cpp:client",
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:jdk-support",
        "//src/main/tools:linux-sandbox",
        "//tools/osx:xcode-locator",
        ":java-version",
    ],
    outs = ["package" + suffix + ".zip"],
    cmd = "$(location :package-bazel.sh) $@ " + ("" if embed else "''") + " $(SRCS)",
    tools = ["package-bazel.sh"],
) for suffix, embed in [
    ("", True),
    ("_notools", False),
    ("_with_jdk", True),
]]

[genrule(
    name = "bazel-bin" + suffix,
    srcs = [
        "//src/main/cpp:client",
        "package-zip" + suffix,
    ],
    outs = ["bazel" + suffix],
    # In msys, a file path without .exe suffix(say foo), refers to a file with .exe
    # suffix(say foo.exe), if foo.exe exists and foo doesn't. So, on windows, we
    # need to remove bazel.exe first, so that cat to bazel won't fail.
    cmd = "rm -f $@; cat $(location //src/main/cpp:client) $(location :package-zip" + suffix + ") > $@ && zip -qA $@",
    executable = 1,
    output_to_bindir = 1,
    visibility = [
        "//scripts:__pkg__",  # For bash completion generation
        "//scripts/packages:__pkg__",  # For installer generation
        "//src/java:__subpackages__",  # For command line reference generation
        "//src/test:__subpackages__",  # For integration tests
    ],
) for suffix in [
    "",
    "_notools",
    "_with_jdk",
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
    "_with_jdk",
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

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_msys",
    values = {"cpu": "x64_windows_msys"},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/java_tools/buildjar:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/coverage:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/junit4:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/testing:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util:srcs",
        "//src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:srcs",
        "//src/java_tools/junitrunner/javatests/com/google/testing/junit/runner/testbed:srcs",
        "//src/java_tools/singlejar:srcs",
        "//src/main/cpp:srcs",
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib:srcs",
        "//src/main/java/com/google/devtools/build/lib/causes:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/common/options:srcs",
        "//src/main/java/com/google/devtools/skylark:srcs",
        "//src/main/native:srcs",
        "//src/main/protobuf:srcs",
        "//src/main/tools:srcs",
        "//src/objc_tools/bundlemerge:srcs",
        "//src/objc_tools/plmerge:srcs",
        "//src/objc_tools/xcodegen:srcs",
        "//src/test/cpp:srcs",
        "//src/test/gen:srcs",
        "//src/test/native:srcs",
        "//src/test/java/com/google/devtools/build/android:srcs",
        "//src/test/java/com/google/devtools/build/docgen:srcs",
        "//src/test/java/com/google/devtools/build/lib:srcs",
        "//src/test/java/com/google/devtools/build/skyframe:srcs",
        "//src/test/java/com/google/devtools/common/options:srcs",
        "//src/test/shell:srcs",
        "//src/tools/android/java/com/google/devtools/build/android:srcs",
        "//src/tools/benchmark:srcs",
        "//src/tools/generate_workspace:srcs",
        "//src/tools/xcode/actoolwrapper:srcs",
        "//src/tools/xcode/environment:srcs",
        "//src/tools/xcode/ibtoolwrapper:srcs",
        "//src/tools/xcode/momcwrapper:srcs",
        "//src/tools/xcode/realpath:srcs",
        "//src/tools/singlejar:srcs",
        "//src/tools/xcode/stdredirect:srcs",
        "//src/tools/xcode/swiftstdlibtoolwrapper:srcs",
        "//src/tools/xcode-common:srcs",
        "//src/tools/remote_worker:srcs",
        "//tools/osx:srcs",
    ],
    visibility = ["//:__pkg__"],
)

genrule(
    name = "derived_java_srcs",
    srcs = [
        "//src/main/protobuf:dist_jars",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:dist_jars",
    ],
    outs = ["derived_java_srcs.zip"],
    cmd = "$(location :combine_derived_java_srcs.sh) $@ $(SRCS)",
    tools = ["combine_derived_java_srcs.sh"],
    visibility = ["//:__pkg__"],
)
