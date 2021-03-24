load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "inconsistent_filesystem_exception",
    srcs = ["InconsistentFilesystemException.java"],
)

java_library(
    name = "file_symlink_exception",
    srcs = ["FileSymlinkException.java"],
)

java_library(
    name = "file_symlink_cycle_exception",
    srcs = ["FileSymlinkCycleException.java"],
    deps = [
        ":file_symlink_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "abstract_chain_uniqueness_function",
    srcs = ["AbstractFileChainUniquenessFunction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_symlink_cycle_uniqueness_function",
    srcs = ["FileSymlinkCycleUniquenessFunction.java"],
    deps = [
        ":abstract_chain_uniqueness_function",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_symlink_infinite_expansion_exception",
    srcs = ["FileSymlinkInfiniteExpansionException.java"],
    deps = [
        ":file_symlink_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_symlink_infinite_expansion_uniqueness_function",
    srcs = ["FileSymlinkInfiniteExpansionUniquenessFunction.java"],
    deps = [
        ":abstract_chain_uniqueness_function",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)
