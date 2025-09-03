load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "DecompressorTests_lib",
    srcs = glob(["*.java"]),
    data = [
        "test_decompress_archive.tar.gz",
        "test_decompress_archive.zip",
        "test_files.ar",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:exception",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/decompressor",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:java-diff-utils",
        "//third_party:junit4",
        "//third_party:truth",
        "@rules_java//java/runfiles",
    ],
)

java_test(
    name = "DecompressorTests",
    tags = [
        "rules",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":DecompressorTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
