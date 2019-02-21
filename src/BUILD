# Packaging

load(":embedded_tools.bzl", "srcsfile")
load(":rule_size_test.bzl", "rule_size_test")

exports_files(["jdeps_modules.golden"])

md5_cmd = "set -e -o pipefail && cat $(SRCS) | sort | %s | awk '{ print $$1; }' > $@"

# TODO(bazel-team): find a better way to handle dylib extensions.
filegroup(
    name = "libunix",
    srcs = select({
        "//src/conditions:darwin": ["//src/main/native:libunix.dylib"],
        "//src/conditions:darwin_x86_64": ["//src/main/native:libunix.dylib"],
        "//src/conditions:windows": ["//src/main/native/windows:windows_jni.dll"],
        "//conditions:default": ["//src/main/native:libunix.so"],
    }),
    visibility = [
        "//src/test/java:__pkg__",
        "//src/tools/remote:__subpackages__",
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
    ] + embedded_tools_target,
    outs = ["install_base_key" + suffix],
    cmd = select({
        "//src/conditions:darwin": md5_cmd % "/sbin/md5",
        "//src/conditions:darwin_x86_64": md5_cmd % "/sbin/md5",
        "//src/conditions:freebsd": md5_cmd % "/sbin/md5",
        "//conditions:default": md5_cmd % "md5sum",
    }),
) for suffix, embedded_tools_target in {
    "_jdk_allmodules": [":embedded_tools_jdk_allmodules"],
    "_jdk_minimal": [":embedded_tools_jdk_minimal"],
    "_dev_jdk": [":embedded_tools_dev_jdk"],
    "_nojdk": [":embedded_tools_nojdk"],
    "_notools": [],
}.items()]

# Try to grab the java version from the java_toolchain.
# Unfortunately, we don't have access to the javac options
# so we cannot get it from the command-line options.
genquery(
    name = "java_toolchain_content",
    expression = "kind(java_toolchain, deps(@bazel_tools//tools/jdk:current_java_toolchain))",
    opts = ["--output=xml"],
    scope = ["@bazel_tools//tools/jdk:current_java_toolchain"],
)

# Create dummy tools so we can do select to prevent building iOS target on
# Linux.
OSX_DUMMY_TARGETS = [
    "src/tools/xcode/actoolwrapper/actoolwrapper",
    "src/tools/xcode/ibtoolwrapper/ibtoolwrapper",
    "src/tools/xcode/momcwrapper/momcwrapper",
    "src/objc_tools/bundlemerge/bundlemerge_deploy.jar",
    "src/objc_tools/plmerge/plmerge_deploy.jar",
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
        "//src/objc_tools/bundlemerge:bundlemerge_deploy.jar",
        "//src/objc_tools/plmerge:plmerge_deploy.jar",
        "//src/tools/xcode/realpath:realpath",
        "//tools/osx:xcode_locator.m",
    ] + select({
        "//src/conditions:iphonesdk": ["//src/tools/xcode/stdredirect:StdRedirect.dylib"],
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

py_library(
    name = "create_embedded_tools_lib",
    srcs = ["create_embedded_tools_lib.py"],
    visibility = ["//:__pkg__"],
)

py_binary(
    name = "create_embedded_tools",
    srcs = ["create_embedded_tools.py"],
    deps = [":create_embedded_tools_lib"],
)

# The tools Bazel uses to compile Java.
# TODO(#6316): Gradually remove the targets here.
JAVA_TOOLS = [
    "//third_party/java/jdk/langtools:test-srcs",
    "//third_party/java/proguard:embedded_tools",
    "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:srcs",
    "//src/tools/singlejar:embedded_tools",
    "//src/java_tools/import_deps_checker/java/com/google/devtools/build/importdeps:embedded_tools",
    "//third_party/ijar",
    "//third_party/ijar:zipper",
    "//third_party/ijar:embedded_tools",
    "//third_party/java/jacoco:srcs",
] + select({
    "//src/conditions:arm": [],
    "//conditions:default": [
        "//src/tools/singlejar:singlejar_local",
    ],
})

# TODO(philwo): Clean this up, once a Bazel that can run py_binary and sh_binary
# in Skylark rules on Windows has been released.
[filegroup(
    name = "embedded_tools" + suffix + "_srcs",
    srcs = JAVA_TOOLS + [
               "BUILD.tools",
               "//tools:embedded_tools_srcs",
               "//third_party:gpl-srcs",
               "//third_party/java/java_tools:bazel_embedded_java_tools",
               "//third_party/java/j2objc:embedded_tools_srcs",
               "//third_party/py/concurrent:srcs",
               "//third_party/py/gflags:srcs",
               "//third_party/py/six:srcs",
               "//src/conditions:embedded_tools",
               "//src/tools/android/java/com/google/devtools/build/android:embedded_tools",
               "//src/tools/launcher:srcs",
               "//src/main/cpp/util:embedded_tools",
               "//src/main/native:embedded_tools",
               "//src/main/protobuf:srcs",
               "//third_party/def_parser:srcs",
               "//third_party/zlib:embedded_tools",
           ] + select({
               "//src/conditions:darwin": [
                   ":darwin_tools",
               ],
               "//src/conditions:darwin_x86_64": [
                   ":darwin_tools",
               ],
               "//conditions:default": [
                   ":dummy_darwin_tools",
               ],
           }) + select({
               "//src/conditions:windows": [
                   "//src/tools/launcher:launcher",
                   "//third_party/def_parser:def_parser",
               ],
               "//conditions:default": [],
           }) +
           jdk,
    visibility = [
        "//src/test:__subpackages__",  # For integration tests
    ],
) for suffix, jdk in {
    "_jdk_allmodules": [":embedded_jdk_allmodules_cached"],
    "_jdk_minimal": [":embedded_jdk_minimal_cached"],
    "_dev_jdk": [":embedded_jdk_minimal"],
    "_nojdk": [],
}.items()]

rule_size_test(
    name = "embedded_tools_size_test",
    src = ":embedded_tools_jdk_allmodules_srcs",
    # WARNING: Only adjust the number in `expect` if you are intentionally
    # adding or removing embedded tools. Know that the more embedded tools there
    # are in Bazel, the bigger the binary becomes and the slower Bazel starts.
    expect = 503,
    margin = 5,  # percentage
)

filegroup(
    name = "embedded_jdk_allmodules_cached",
    srcs = select({
        "//src/conditions:darwin": [
            "@openjdk_macos//file",
        ],
        "//src/conditions:darwin_x86_64": [
            "@openjdk_macos//file",
        ],
        "//src/conditions:windows": [
            "@openjdk_win//file",
        ],
        "//src/conditions:linux_aarch64": [
            "@openjdk_linux_aarch64//file",
        ],
        "//conditions:default": [
            "@openjdk_linux//file",
        ],
    }),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "embedded_jdk_minimal_cached",
    srcs = select({
        "//src/conditions:darwin": [
            "@openjdk_macos_minimal//file",
        ],
        "//src/conditions:darwin_x86_64": [
            "@openjdk_macos_minimal//file",
        ],
        "//src/conditions:windows": [
            "@openjdk_win_minimal//file",
        ],
        "//conditions:default": [
            "@openjdk_linux_minimal//file",
        ],
    }),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "embedded_jdk_vanilla",
    srcs = select({
        "//src/conditions:darwin": [
            "@openjdk_macos_vanilla//file",
        ],
        "//src/conditions:darwin_x86_64": [
            "@openjdk_macos_vanilla//file",
        ],
        "//src/conditions:windows": [
            "@openjdk_win_vanilla//file",
        ],
        "//src/conditions:linux_aarch64": [
            "@openjdk_linux_aarch64_vanilla//file",
        ],
        "//conditions:default": [
            "@openjdk_linux_vanilla//file",
        ],
    }),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

sh_binary(
    name = "minimize_jdk",
    srcs = ["minimize_jdk.sh"],
)

filegroup(
    name = "embedded_jdk_minimal",
    srcs = select({
        "//src/conditions:windows": [
            ":embedded_jdk_minimal.zip",
        ],
        "//conditions:default": [
            ":embedded_jdk_minimal.tar.gz",
        ],
    }),
)

filegroup(
    name = "embedded_jdk_allmodules",
    srcs = select({
        "//src/conditions:windows": [
            ":embedded_jdk_allmodules.zip",
        ],
        "//conditions:default": [
            ":embedded_jdk_allmodules.tar.gz",
        ],
    }),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

[genrule(
    name = "embedded_jdk_minimal" + suffix,
    srcs = [
        ":embedded_jdk_vanilla",
        ":minimize_jdk.sh",
        ":jdeps_modules.golden",
    ],
    outs = ["minimal_jdk" + suffix],
    cmd = "$(location :minimize_jdk.sh) $(location :embedded_jdk_vanilla) $(location :jdeps_modules.golden) $(OUTS)",
    visibility = ["//src/test/shell/bazel:__pkg__"],
) for suffix in [
    ".zip",
    ".tar.gz",
]]

[genrule(
    name = "embedded_jdk_allmodules" + suffix,
    srcs = [
        ":embedded_jdk_vanilla",
        ":minimize_jdk.sh",
        ":jdeps_modules.golden",
    ],
    outs = ["allmodules_jdk" + suffix],
    cmd = "$(location :minimize_jdk.sh) --allmodules $(location :embedded_jdk_vanilla) $(location :jdeps_modules.golden) $(OUTS)",
    visibility = ["//src/test/shell/bazel:__pkg__"],
) for suffix in [
    ".zip",
    ".tar.gz",
]]

[srcsfile(
    name = "embedded_tools" + suffix + "_params",
    srcs = [":embedded_tools" + suffix + "_srcs"],
    out = "embedded_tools" + suffix + ".params",
) for suffix in [
    "_jdk_allmodules",
    "_jdk_minimal",
    "_dev_jdk",
    "_nojdk",
]]

[genrule(
    name = "embedded_tools" + suffix,
    srcs = [
        ":embedded_tools" + suffix + "_params",
        ":embedded_tools" + suffix + "_srcs",
    ],
    outs = ["embedded_tools" + suffix + ".zip"],
    cmd = "$(location :create_embedded_tools) \"$@\" $(location :embedded_tools" + suffix + "_params)",
    tools = [":create_embedded_tools"],
) for suffix in [
    "_jdk_allmodules",
    "_jdk_minimal",
    "_dev_jdk",
    "_nojdk",
]]

[genrule(
    name = "package-zip" + suffix,
    srcs = ([":embedded_tools" + suffix + ".zip"] if embed else []) + [
        # The script assumes that the embedded tools zip (if exists) is the
        # first item here, the deploy jar the second and install base key is the
        # third
        "//src/main/java/com/google/devtools/build/lib:bazel/BazelServer_deploy.jar",
        "install_base_key" + suffix,
        ":libunix",
        "//src/main/tools:build-runfiles",
        "//src/main/tools:process-wrapper",
        "//src/main/tools:jdk-support",
        "//src/main/tools:linux-sandbox",
        "//tools/osx:xcode-locator",
    ] + select({
        "//src/conditions:windows": [],
        "//conditions:default": [
            "//src/main/tools:daemonize",
        ],
    }),
    outs = ["package" + suffix + ".zip"],
    cmd = "$(location :package-bazel.sh) $@ " + ("" if embed else "''") + " $(SRCS)",
    tools = ["package-bazel.sh"],
) for suffix, embed in [
    ("_jdk_allmodules", True),
    ("_jdk_minimal", True),
    ("_dev_jdk", True),
    ("_notools", False),
    ("_nojdk", True),
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
    "_jdk_allmodules",
    "_jdk_minimal",
    "_dev_jdk",
    "_notools",
    "_nojdk",
]]

[genrule(
    name = "bazel-bin" + suffix,
    srcs = [":bazel-bin" + jdk],
    outs = ["bazel" + suffix],
    # In msys, a file path without .exe suffix(say foo), refers to a file with .exe
    # suffix(say foo.exe), if foo.exe exists and foo doesn't. So, on windows, we
    # need to remove bazel.exe first, so that cat to bazel won't fail.
    cmd = "rm -f $@; cp $(location :bazel-bin" + jdk + ") $@",
    executable = 1,
    output_to_bindir = 1,
    visibility = [
        "//scripts:__pkg__",  # For bash completion generation
        "//scripts/packages:__pkg__",  # For installer generation
        "//src/java:__subpackages__",  # For command line reference generation
        "//src/test:__subpackages__",  # For integration tests
    ],
) for suffix, jdk in [
    ("-dev", "_jdk_allmodules"),
    ("", "_jdk_minimal"),
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
    "_nojdk",
]]

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/conditions:srcs",
        "//src/java_tools/buildjar:srcs",
        "//src/java_tools/import_deps_checker:srcs",
        "//src/java_tools/junitrunner:srcs",
        "//src/java_tools/singlejar:srcs",
        "//src/main/cpp:srcs",
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib:srcs",
        "//src/main/java/com/google/devtools/build/skydoc:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/common/options:srcs",
        "//src/main/java/com/google/devtools/starlark:srcs",
        "//src/main/native:srcs",
        "//src/main/protobuf:srcs",
        "//src/main/tools:srcs",
        "//src/objc_tools/bundlemerge:srcs",
        "//src/objc_tools/plmerge:srcs",
        "//src/test/cpp:srcs",
        "//src/test/gen:srcs",
        "//src/test/native:srcs",
        "//src/test/starlark:srcs",
        "//src/test/java/com/google/devtools/build/android:srcs",
        "//src/test/java/com/google/devtools/build/docgen:srcs",
        "//src/test/java/com/google/devtools/build/lib:srcs",
        "//src/test/java/com/google/devtools/build/lib/shell:srcs",
        "//src/test/java/com/google/devtools/build/skydoc:srcs",
        "//src/test/java/com/google/devtools/build/skyframe:srcs",
        "//src/test/java/com/google/devtools/common/options:srcs",
        "//src/test/py/bazel:srcs",
        "//src/test/shell:srcs",
        "//src/tools/android/java/com/google/devtools/build/android:srcs",
        "//src/tools/execlog:srcs",
        "//src/tools/workspacelog:srcs",
        "//src/tools/launcher:srcs",
        "//src/tools/package_printer/java/com/google/devtools/build/packageprinter:srcs",
        "//src/tools/skylark/java/com/google/devtools/skylark/common:srcs",
        "//src/tools/skylark/java/com/google/devtools/skylark/skylint:srcs",
        "//src/tools/skylark/javatests/com/google/devtools/skylark/skylint:srcs",
        "//src/tools/xcode/actoolwrapper:srcs",
        "//src/tools/xcode/ibtoolwrapper:srcs",
        "//src/tools/xcode/momcwrapper:srcs",
        "//src/tools/xcode/realpath:srcs",
        "//src/tools/singlejar:srcs",
        "//src/tools/xcode/stdredirect:srcs",
        "//src/tools/xcode-common:srcs",
        "//src/tools/remote:srcs",
        "//tools/osx:srcs",
    ],
    visibility = ["//:__pkg__"],
)

genrule(
    name = "derived_java_srcs",
    srcs = [
        "//src/main/protobuf:dist_jars",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:dist_jars",
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:dist_jars",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:dist_jars",
        "@googleapis//:dist_jars",
        "@remoteapis//:dist_jars",
    ],
    outs = ["derived_java_srcs.zip"],
    cmd = "$(location :combine_derived_java_srcs.sh) $(JAVABASE) $@ $(SRCS)",
    toolchains = ["@bazel_tools//tools/jdk:current_host_java_runtime"],
    tools = ["combine_derived_java_srcs.sh"],
    visibility = ["//:__pkg__"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/test/cpp:all_windows_tests",
        "//src/test/java/com/google/devtools/build/android:all_windows_tests",
        "//src/test/java/com/google/devtools/build/docgen:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib:all_windows_tests",
        "//src/test/java/com/google/devtools/build/skyframe:all_windows_tests",
        "//src/test/java/com/google/devtools/common/options:all_windows_tests",
        "//src/test/native:all_windows_tests",
        "//src/test/py/bazel:all_windows_tests",
        "//src/test/shell:all_windows_tests",
        "//src/tools/launcher:all_windows_tests",
        "//src/tools/singlejar:all_windows_tests",
        "//third_party/def_parser:all_windows_tests",
        "//tools:all_windows_tests",
    ],
)

# config_setting alias left here because bazel-only third_party code is pointing to it
alias(
    name = "windows",
    actual = "//src/conditions:windows",
    visibility = ["//visibility:public"],
)
