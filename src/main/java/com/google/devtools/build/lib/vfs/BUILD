load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

OS_PATH_POLICY_SOURCES = [
    "OsPathPolicy.java",
    "UnixOsPathPolicy.java",
    "WindowsOsPathPolicy.java",
]

PATH_FRAGMENT_SOURCES = [
    "PathFragment.java",
    "PathSegmentIterator.java",
    "PathStrippable.java",
]

OUTPUT_SERVICE_SOURCES = [
    "OutputService.java",
    "LocalOutputService.java",
]

# Separate from :vfs since it refers to repositories by name, which :cmdline defines on top of :vfs.
REWINDABLE_REPO_FILE_SYSTEM_SOURCES = ["RewindableRepoFileSystem.java"]

java_library(
    name = "ospathpolicy",
    srcs = OS_PATH_POLICY_SOURCES,
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/windows:windows_path_operations",
        "//third_party:guava",
    ],
)

java_library(
    name = "pathfragment",
    srcs = PATH_FRAGMENT_SOURCES,
    deps = [
        ":ospathpolicy",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "vfs",
    srcs = glob(
        [
            "*.java",
        ],
        exclude = PATH_FRAGMENT_SOURCES + OUTPUT_SERVICE_SOURCES + OS_PATH_POLICY_SOURCES +
                  REWINDABLE_REPO_FILE_SYSTEM_SOURCES,
    ),
    deps = [
        ":pathfragment",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/io:file_symlink_exception",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:file_channels",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "rewindable_repo_file_system",
    srcs = REWINDABLE_REPO_FILE_SYSTEM_SOURCES,
    deps = [
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "output_service",
    srcs = OUTPUT_SERVICE_SOURCES,
    deps = [
        ":pathfragment",
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
