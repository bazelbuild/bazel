package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/serializers:srcs",
    ],
)

java_library(
    name = "serialization",
    srcs = glob(
        ["**/*.java"],
        exclude = ["SerializationConstants.java"],
    ),
    exports = [
        ":constants",
    ],
    deps = [
        ":kryo",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:registered-singleton",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:unsafe-provider",
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

# A convenience library that bundles together kryo and its runtime dependencies.
java_library(
    name = "kryo",
    exports = [
        "//third_party:kryo",
        "//third_party:objenesis",
    ],
    runtime_deps = [
        "//third_party:minlog",
        "//third_party:reflectasm",
    ],
)
