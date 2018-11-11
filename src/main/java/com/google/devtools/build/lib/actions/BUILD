package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

LOCALHOST_CAPACITY_SRCS = [
    "LocalHostCapacity.java",
    "LocalHostResourceManagerLinux.java",
    "LocalHostResourceManagerDarwin.java",
    "LocalHostResourceFallback.java",
    "ResourceSet.java",
]

# Action graph and cache.
java_library(
    name = "actions",
    srcs = glob(
        [
            "*.java",
            "cache/*.java",
        ],
        exclude = ["CommandLineItem.java"] + LOCALHOST_CAPACITY_SRCS,
    ),
    deps = [
        ":commandline_item",
        ":localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib:command-utils",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:fingerprint_cache",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:action_cache_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "commandline_item",
    srcs = ["CommandLineItem.java"],
)

java_library(
    name = "localhost_capacity",
    srcs = LOCALHOST_CAPACITY_SRCS,
    deps = [
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)
