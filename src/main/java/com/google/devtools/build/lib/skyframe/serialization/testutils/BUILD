package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "testutils",
    testonly = 1,
    srcs = glob(
        ["*.java"],
        exclude = [
            "FakeDirectories.java",
            "SerializationDepsUtils.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
        # TODO(https://github.com/bazelbuild/bazel/issues/5985): Added to suppress warnings.
        "//src/main/java/com/google/devtools/build/lib:skylark-semantics",
    ],
)

java_library(
    name = "depsutils",
    srcs = ["SerializationDepsUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
        # TODO(https://github.com/bazelbuild/bazel/issues/5985): Added to suppress warnings.
        "//src/main/java/com/google/devtools/build/lib:skylark-semantics",
    ],
)

java_library(
    name = "fake_directories",
    testonly = 1,
    srcs = ["FakeDirectories.java"],
    deps = [
        ":testutils",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)
