package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

PATH_FRAGMENT_SOURCES = [
    "PathFragment.java",
    "PathFragmentSerializationProxy.java",
    "OsPathPolicy.java",
    "UnixOsPathPolicy.java",
    "WindowsOsPathPolicy.java",
]

OUTPUT_SERVICE_SOURCES = [
    "OutputService.java",
]

java_library(
    name = "pathfragment",
    srcs = PATH_FRAGMENT_SOURCES,
    deps = [
        "//src/main/java/com/google/devtools/build/lib:filetype",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/windows:windows_short_path",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//third_party:guava",
    ],
)

# Virtual file system; do not use externally!
java_library(
    name = "vfs",
    srcs = glob(
        [
            "*.java",
        ],
        exclude = PATH_FRAGMENT_SOURCES + OUTPUT_SERVICE_SOURCES,
    ),
    visibility = ["//visibility:public"],
    exports = [
        ":pathfragment",
    ],
    deps = [
        ":pathfragment",
        "//src/main/java/com/google/devtools/build/lib:filetype",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "output_service",
    srcs = OUTPUT_SERVICE_SOURCES,
    deps = [
        ":pathfragment",
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
