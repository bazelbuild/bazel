# This should correspond to the list of "EMBEDDED_TOOLS" in TestConstants.java.bazel.
filegroup(
    name = "embedded_scripts",
    testonly = 1,
    data = [
        "//src/main/tools:build-runfiles",
        "//src/main/tools:build_interface_so",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:process-wrapper",
    ],
)

java_library(
    name = "testutil",
    srcs = glob(["com/google/devtools/build/lib/testutil/*.java"]),
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:os_util",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "options_test",
    srcs = glob(["com/google/devtools/common/options/*.java"]),
    args = ["com.google.devtools.common.options.AllTests"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_prefix = "com/google/devtools/build/lib"

java_library(
    name = "foundations_testutil",
    srcs = glob([
        "com/google/devtools/build/lib/vfs/util/*.java",
        "com/google/devtools/build/lib/events/util/*.java",
    ]),
    data = [
        "//src/main/native:libunix.dylib",
        "//src/main/native:libunix.so",
    ],
    deps = [
        ":testutil",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:packages",
        "//src/main/java:shell",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "test_runner",
    srcs = [test_prefix + "/AllTests.java"],
    deps = [
        ":testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "foundations_test",
    srcs = glob(
        ["%s/%s" % (test_prefix, p) for p in [
            "cmdline/*.java",
            "concurrent/*.java",
            "collect/*.java",
            "collect/nestedset/*.java",
            "events/*.java",
            "testutiltests/*.java",
            "unix/*.java",
            "util/*.java",
            "util/io/*.java",
            "vfs/*.java",
            "vfs/inmemoryfs/*.java",
        ]],
        # java_rules_skylark doesn't support resource loading with
        # qualified paths.
        exclude = [
            test_prefix + f
            for f in [
                "/util/DependencySetWindowsTest.java",
                "/util/ResourceFileLoaderTest.java",
                "/vfs/PathFragmentWindowsTest.java",
                "/vfs/PathWindowsTest.java",
            ]
        ],
    ),
    args = ["com.google.devtools.build.lib.AllTests"],
    data = glob([test_prefix + "/vfs/*.zip"]),
    tags = ["foundations"],
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:base-util",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:os_util",
        "//src/main/java:packages",
        "//src/main/java:shell",
        "//src/main/java:unix",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "windows_test",
    srcs = glob(["%s/%s" % (test_prefix, p) for p in [
        "util/DependencySetWindowsTest.java",
        "vfs/PathFragmentWindowsTest.java",
        "vfs/PathWindowsTest.java",
    ]]),
    args = [
        "com.google.devtools.build.lib.AllTests",
    ],
    jvm_flags = ["-Dblaze.os=Windows"],
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "actions_testutil",
    srcs = glob([
        "com/google/devtools/build/lib/actions/util/*.java",
    ]),
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:packages",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "actions_test",
    srcs = glob([
        "com/google/devtools/build/lib/actions/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    deps = [
        ":actions_testutil",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_library(
    name = "analysis_testutil",
    srcs = glob([
        "com/google/devtools/build/lib/analysis/mock/BazelAnalysisMock.java",
        "com/google/devtools/build/lib/analysis/util/*.java",
        "com/google/devtools/build/lib/exec/util/*.java",
        "com/google/devtools/build/lib/skyframe/util/*.java",
    ]),
    resources = [
        "MOCK_CROSSTOOL",
    ],
    tags = ["analysis"],
    deps = [
        ":actions_testutil",
        ":foundations_testutil",
        ":packages_testutil",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:buildtool-runtime",
        "//src/main/java:clock",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:packages",
        "//src/main/java:runtime",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "analysis_actions_test",
    srcs = glob([
        "com/google/devtools/build/lib/analysis/actions/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["analysis"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:io",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/protobuf:extra_actions_base_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "analysis_config_test",
    srcs = glob([
        "com/google/devtools/build/lib/analysis/config/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["analysis"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:common",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "analysis_constraints_test",
    srcs = glob([
        "com/google/devtools/build/lib/analysis/constraints/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["analysis"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "analysis_select_test",
    srcs = glob([
        "com/google/devtools/build/lib/analysis/select/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["analysis"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "analysis_test",
    srcs = glob([
        "com/google/devtools/build/lib/analysis/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["analysis"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_library(
    name = "packages_testutil",
    srcs = glob([
        "com/google/devtools/build/lib/packages/util/*.java",
    ]),
    deps = [
        ":foundations_testutil",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:clock",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:packages",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "packages_test",
    srcs = glob([
        "com/google/devtools/build/lib/packages/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ideinfo_test",
    srcs = glob([
        "com/google/devtools/build/lib/ideinfo/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["ideinfo"],
    deps = [
        ":analysis_testutil",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/protobuf:android_studio_ide_info_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

cc_binary(
    name = "com/google/devtools/build/lib/shell/killmyself",
    srcs = ["com/google/devtools/build/lib/shell/killmyself.cc"],
)

java_test(
    name = "shell_test",
    srcs = glob([
        "com/google/devtools/build/lib/shell/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    data = [
        ":com/google/devtools/build/lib/shell/killmyself",
    ],
    tags = ["shell"],
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:shell",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_binary(
    name = "skylarkshell",
    srcs = ["com/google/devtools/build/lib/syntax/SkylarkShell.java"],
    main_class = "com.google.devtools.build.lib.syntax.SkylarkShell",
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:events",
        "//src/main/java:packages",
    ],
)

java_test(
    name = "syntax_test",
    srcs = glob([
        "com/google/devtools/build/lib/syntax/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    deps = [
        ":foundations_testutil",
        ":syntax_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "syntax_testutil",
    srcs = glob([
        "com/google/devtools/build/lib/syntax/util/*.java",
    ]),
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "skylark_testutil",
    srcs = glob([
        "com/google/devtools/build/lib/skylark/util/*.java",
    ]),
    deps = [
        ":analysis_testutil",
        ":foundations_testutil",
        ":syntax_testutil",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "skylark_test",
    srcs = glob([
        "com/google/devtools/build/lib/skylark/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":skylark_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ziputils-tests",
    srcs = glob(["com/google/devtools/build/android/ziputils/*.java"]),
    args = ["com.google.devtools.build.android.ziputils.AllTests"],
    tags = ["ziputils"],
    deps = [
        ":testutil",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:splitter_lib",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:ziputils_lib",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "filegroup-rules-tests",
    srcs = glob(["com/google/devtools/build/lib/rules/filegroup/*.java"]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["rules"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "java-rules-tests",
    srcs = glob(["com/google/devtools/build/lib/rules/java/*.java"]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["rules"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "repository_test",
    srcs = glob([
        "com/google/devtools/build/lib/bazel/repository/*.java",
    ]),
    args = ["com.google.devtools.build.lib.AllTests"],
    tags = ["rules"],
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:bazel-core",
        "//src/main/java:collect",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "runtime-tests",
    srcs = glob(["com/google/devtools/build/lib/runtime/*.java"]),
    args = ["com.google.devtools.build.lib.AllTests"],
    deps = [
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:buildtool-runtime",
        "//src/main/java:clock",
        "//src/main/java:io",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:invocation_policy_proto",
        "//src/main/protobuf:test_status_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "sandbox-tests",
    srcs = glob(["com/google/devtools/build/lib/sandbox/*.java"]),
    args = ["com.google.devtools.build.lib.AllTests"],
    data = [":embedded_scripts"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":testutil",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:clock",
        "//src/main/java:events",
        "//src/main/java:os_util",
        "//src/main/java:shell",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "worker-example-lib",
    srcs = glob(["com/google/devtools/build/lib/worker/ExampleWorker*.java"]),
    visibility = [
        "//src/test/shell/bazel:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_proto",
        "//third_party:guava",
    ],
)

java_binary(
    name = "worker-example",
    main_class = "com.google.devtools.build.lib.worker.ExampleWorker",
    visibility = [
        "//src/test/shell/bazel:__pkg__",
    ],
    runtime_deps = [
        ":worker-example-lib",
    ],
)

TEST_SUITES = [
    "ziputils",
    "rules",
    "analysis",
    "foundations",
    "shell",
    "ideinfo",
]

[
    test_suite(
        name = n,
        tags = [n],
    )
    for n in TEST_SUITES
]

test_suite(
    name = "others",
    tags = ["-" + n for n in TEST_SUITES],
)
