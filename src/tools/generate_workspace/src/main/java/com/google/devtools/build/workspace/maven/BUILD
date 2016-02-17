java_library(
    name = "maven",
    srcs = [
        "DefaultModelResolver.java",
        "Resolver.java",
    ],
    visibility = [
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace:__pkg__",
        "//src/tools/generate_workspace/src/test:__subpackages__",
    ],
    runtime_deps = [
        "//third_party:maven_model",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
    ],
    deps = [
        ":rule",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:maven-connector",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
    ],
)

java_library(
    name = "rule",
    srcs = ["Rule.java"],
    visibility = [
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace:__pkg__",
        "//src/tools/generate_workspace/src/test:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:maven-connector",
        "//third_party:aether",
        "//third_party:guava",
        "//third_party:maven_model",
    ],
)
