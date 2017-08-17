# TODO(janakr): find out how to avoid this default visibility and still have
# automatic BUILD-file generation.
package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "serialization",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
