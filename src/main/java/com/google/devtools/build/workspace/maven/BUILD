java_library(
    name = "maven",
    srcs = [
        "DefaultModelResolver.java",
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
        "//src/main/java:maven-connector",
        "//third_party:guava",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
    ],
)

java_library(
    name = "rule",
    srcs = ["Rule.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build:__subpackages__",
    ],
    deps = [
        "//third_party:aether",
        "//third_party:guava",
    ],
)
