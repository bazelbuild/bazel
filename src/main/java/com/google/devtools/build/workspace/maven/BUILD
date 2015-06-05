java_library(
    name = "maven",
    srcs = [
        "Resolver.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/workspace:__pkg__",
        "//src/test/java/com/google/devtools/build/workspace:__pkg__",
    ],
    runtime_deps = [
        "//third_party:maven_model",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
    ],
    deps = [
        ":rule",
        "//src/main/java:events",
        "//third_party:aether",
        "//third_party:guava",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
    ],
)

java_library(
    name = "rule",
    srcs = ["Rule.java"],
    deps = ["//third_party:auto_value"],
)
