load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/vfs/bazel:srcs",
        "//src/test/java/com/google/devtools/build/lib/vfs/util:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

# Tests for Windows-specific functionality that can run cross-platform.
CROSS_PLATFORM_WINDOWS_TESTS = [
    "PathFragmentWindowsTest.java",
    "WindowsPathTest.java",
]

# Tests for Windows-specific functionality that run on Windows.
WINDOWS_ON_WINDOWS_TESTS = [
]

# All Windows-specific tests. Use this to exclude Windows tests from globs.
ALL_WINDOWS_TESTS = CROSS_PLATFORM_WINDOWS_TESTS + WINDOWS_ON_WINDOWS_TESTS

java_library(
    name = "SymlinkAwareFileSystemTest",
    srcs = ["SymlinkAwareFileSystemTest.java"],
    deps = [
        ":FileSystemTest_lib",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_library(
    name = "VfsTests_lib",
    srcs = glob(
        [
            "*.java",
            "inmemoryfs/*.java",
        ],
        exclude = ALL_WINDOWS_TESTS + [
            "SymlinkAwareFileSystemTest.java",
            "FileSystemTest.java",
        ],
    ),
    deps = [
        ":SymlinkAwareFileSystemTest",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:round-tripping",
        "//src/main/java/com/google/devtools/build/lib/unix:unix_file_system",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//src/test/java/com/google/devtools/build/lib/vfs/util:test_glob_path_discriminator",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:hash",
        "//third_party/java/guava:testing",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

java_library(
    name = "FileSystemTest_lib",
    srcs = ["FileSystemTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe:default_syscall_cache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "VfsTests",
    size = "large",
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":VfsTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_library(
    name = "VfsWindowsTests_lib",
    srcs = CROSS_PLATFORM_WINDOWS_TESTS + ["PathAbstractTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:ospathpolicy",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:testing",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

# Tests windows specific path handling on Unix.
java_test(
    name = "VfsWindowsTests",
    size = "small",
    jvm_flags = [
        "-Dblaze.os=Windows",
        "-Dbazel.windows_unix_root=C:/fake/msys",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":VfsWindowsTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = ["FileSystemTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe:default_syscall_cache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)
