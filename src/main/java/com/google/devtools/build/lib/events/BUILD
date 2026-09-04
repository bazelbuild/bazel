load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "events",
    srcs = glob(
        ["*.java"],
        exclude = ["EventBusEventHandler.java"],
    ),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/lib:starlark_access",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "event_bus_event_handler",
    srcs = ["EventBusEventHandler.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
    ],
    deps = [
        ":events",
        "//third_party/java/guava:eventbus",
    ],
)
