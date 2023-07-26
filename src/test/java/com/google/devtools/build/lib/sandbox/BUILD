load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = [
        "FakeSandboxfsProcess.java",
        "SandboxedSpawnRunnerTestCase.java",
        "SpawnRunnerTestUtil.java",
    ],
    runtime_deps = [
        "//src/test/shell/integration:spend_cpu_time",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_input_expander",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxfs_process",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
    ],
)

java_library(
    name = "sandboxfs-base-tests",
    testonly = 1,
    srcs = ["BaseSandboxfsProcessIntegrationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxfs_process",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SmallTests",
    size = "small",
    srcs =
        [
            "AbstractContainerizingSandboxedSpawnTest.java",
            "SandboxHelpersTest.java",
            "SandboxOptionsTest.java",
            "SymlinkedSandboxedSpawnTest.java",
        ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:tree_deleter",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandbox_helpers",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandbox_options",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxed_spawns",
        "//src/main/java/com/google/devtools/build/lib/sandbox:tree_deleter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "SandboxFsTests",
    size = "small",
    srcs =
        [
            "BaseRealSandboxfsProcessTest.java",
            "BaseSandboxfsProcessIntegrationTest.java",
            "FakeSandboxfsProcessIntegrationTest.java",
            "RealSandboxfs01ProcessTest.java",
            "RealSandboxfs02ProcessTest.java",
            "SandboxfsSandboxedSpawnTest.java",
        ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandbox_helpers",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxed_spawns",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxfs_process",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxfs_sandboxed_spawn",
        "//src/main/java/com/google/devtools/build/lib/sandbox:tree_deleter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SpawnRunnerTests",
    size = "medium",
    srcs = [
        "DarwinSandboxedSpawnRunnerTest.java",
        "LinuxSandboxCommandLineBuilderTest.java",
        "LinuxSandboxedSpawnRunnerTest.java",
        "ProcessWrapperSandboxedSpawnRunnerTest.java",
    ],
    data = ["//src/test/java/com/google/devtools/build/lib:embedded_scripts"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/exec:tree_deleter",
        "//src/main/java/com/google/devtools/build/lib/sandbox:darwin_sandbox",
        "//src/main/java/com/google/devtools/build/lib/sandbox:linux_sandbox",
        "//src/main/java/com/google/devtools/build/lib/sandbox:linux_sandbox_command_line_builder",
        "//src/main/java/com/google/devtools/build/lib/sandbox:process_wrapper_sandbox",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandbox_helpers",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxed_spawns",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxfs_process",
        "//src/main/java/com/google/devtools/build/lib/sandbox:tree_deleter",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/exec/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "CgroupsInfoTest",
    size = "small",
    srcs = ["CgroupsInfoTest.java"],
    tags = ["no_windows"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox:linux_sandbox",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "sandboxfs-integration-tests",
    srcs = ["RealSandboxfsProcessIntegrationTest.java"],
    data = ["//src/test/java/com/google/devtools/build/lib:embedded_scripts"],
    local = 1,
    tags = [
        "manual",  # Test requires: --test_env=SANDBOXFS=/path/to/sandboxfs
        "no-sandbox",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        ":sandboxfs-base-tests",
        "//src/main/java/com/google/devtools/build/lib/sandbox:sandboxfs_process",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:junit4",
    ],
)
