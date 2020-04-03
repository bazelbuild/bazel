load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
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
    srcs = ["FakeSandboxfsProcess.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "sandboxfs-base-tests",
    testonly = 1,
    srcs = ["BaseSandboxfsProcessIntegrationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SandboxTests",
    size = "medium",
    srcs = glob(
        ["*.java"],
        exclude = [
            "BaseSandboxfsProcessIntegrationTest.java",
            "FakeSandboxfsProcess.java",
            "RealSandboxfsProcessIntegrationTest.java",
        ],
    ),
    data = ["//src/test/java/com/google/devtools/build/lib:embedded_scripts"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        ":sandboxfs-base-tests",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
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
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:junit4",
    ],
)
