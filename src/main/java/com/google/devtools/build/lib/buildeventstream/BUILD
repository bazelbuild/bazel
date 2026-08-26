load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# External build-event reporting infrastructure
java_library(
    name = "buildeventstream",
    srcs = glob(
        ["*.java"],
        exclude = [
            "LargeBuildEventSerializedEvent.java",
            "BuildEventServiceUploadCompleteEvent.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:build_event_service_options",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/netty4:common",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/protobuf/util:time",
    ],
)

java_library(
    name = "build_event_service_upload_complete_event",
    srcs = ["BuildEventServiceUploadCompleteEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "large-buildevent-serialized-event",
    srcs = ["LargeBuildEventSerializedEvent.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety"],
)
