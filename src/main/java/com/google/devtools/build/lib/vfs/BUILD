package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

PATH_FRAGMENT_SOURCES = [
    "Canonicalizer.java",
    "PathFragment.java",
    "PathFragmentSerializationProxy.java",
    "UnixPathFragment.java",
    "WindowsPathFragment.java",
]

java_library(
    name = "pathfragment",
    srcs = PATH_FRAGMENT_SOURCES,
    deps = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:filetype",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
    ],
)

# Virtual file system; do not use externally!
java_library(
    name = "vfs",
    srcs = glob(
        [
            "*.java",
        ],
        exclude = PATH_FRAGMENT_SOURCES,
    ),
    visibility = ["//visibility:public"],
    exports = [
        ":pathfragment",
    ],
    deps = [
        ":pathfragment",
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:filetype",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/windows:windows_short_path",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)
