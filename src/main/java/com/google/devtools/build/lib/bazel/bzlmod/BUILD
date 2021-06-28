load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "common",
    srcs = [
        "ModuleKey.java",
        "RepoSpec.java",
    ],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "registry",
    srcs = [
        "Registry.java",
        "RegistryFactory.java",
    ],
    deps = [
        ":common",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "resolution",
    srcs = [
        "ArchiveOverride.java",
        "DiscoveryFunction.java",
        "DiscoveryValue.java",
        "GitOverride.java",
        "LocalPathOverride.java",
        "Module.java",
        "ModuleFileFunction.java",
        "ModuleFileGlobals.java",
        "ModuleFileValue.java",
        "ModuleOverride.java",
        "NonRegistryOverride.java",
        "ParsedVersion.java",
        "RegistryOverride.java",
        "SelectionFunction.java",
        "SelectionValue.java",
        "SingleVersionOverride.java",
    ],
    deps = [
        ":common",
        ":registry",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/repository",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
