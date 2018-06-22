package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:srcs",
    ],
)

java_library(
    name = "serialization",
    srcs = glob(
        ["**/*.java"],
        exclude = ["SerializationConstants.java"],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:crash-utils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:registered-singleton",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "constants",
    srcs = ["SerializationConstants.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:resource_usage",
    ],
)
