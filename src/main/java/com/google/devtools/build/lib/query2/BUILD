package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "query2",
    srcs = glob(
        ["*.java"],
        exclude = [
            "AbstractBlazeQueryEnvironment.java",
            "FakeLoadTarget.java",
        ],
    ),
    deps = [
        ":abstract-blaze-query-env",
        ":aspect-resolver",
        ":common-query-options",
        ":fake-load-target",
        ":query-engine",
        ":query-output",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:command-utils",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:analysis_java_proto",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "query-output",
    srcs = glob(
        ["output/*.java"],
        exclude = [
            "output/AspectResolver.java",
            "output/PresciseAspectResolver.java",
            "output/ConservativeAspectResolver.java",
            "output/NullAspectResolver.java",
        ],
    ),
    deps = [
        ":abstract-blaze-query-env",
        ":aspect-resolver",
        ":common-query-options",
        ":fake-load-target",
        ":query-engine",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "abstract-blaze-query-env",
    srcs = [
        "AbstractBlazeQueryEnvironment.java",
    ],
    deps = [
        ":query-engine",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "common-query-options",
    srcs = [
        "CommonQueryOptions.java",
    ],
    deps = [
        ":aspect-resolver",
        ":query-engine",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "aspect-resolver",
    srcs = [
        "output/AspectResolver.java",
        "output/ConservativeAspectResolver.java",
        "output/NullAspectResolver.java",
        "output/PreciseAspectResolver.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "fake-load-target",
    srcs = [
        "FakeLoadTarget.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# Query library.
java_library(
    name = "query-engine",
    srcs = glob(["engine/*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
