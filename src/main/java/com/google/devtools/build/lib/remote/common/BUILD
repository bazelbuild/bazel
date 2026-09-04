load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_library(
    name = "action_key",
    srcs = ["ActionKey.java"],
    deps = [
        "//third_party/java/guava:base",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "bulk_transfer_exception",
    srcs = ["BulkTransferException.java"],
    deps = [
        ":cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:important_output_handler",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "cache_not_found_exception",
    srcs = ["CacheNotFoundException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/jsr305_annotations",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "lost_inputs_event",
    srcs = ["LostInputsEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "common",
    srcs = glob(
        ["*.java"],
        exclude = [
            "ActionKey.java",
            "BulkTransferException.java",
            "CacheNotFoundException.java",
            "LostInputsEvent.java",
        ],
    ),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/tools/remote:__subpackages__",
    ],
    deps = [
        ":action_key",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/remote/util:async_task_cache",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:rx_futures",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/rxjava3",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
