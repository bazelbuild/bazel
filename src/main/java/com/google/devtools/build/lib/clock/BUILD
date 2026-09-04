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

# Clock classes.
java_library(
    name = "clock",
    srcs = glob([
        "*.java",
    ]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
        "//third_party/java/guava:annotations",
    ],
)
