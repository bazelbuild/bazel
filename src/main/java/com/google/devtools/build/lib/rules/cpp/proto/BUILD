package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "CcProtoLibrary",
    srcs = [
        "CcProtoAspect.java",
        "CcProtoLibrary.java",
        "CcProtoLibraryRule.java",
    ],
    deps = [
        ":CcProtoLibraryProviders",
        ":ProtoCcHeaderProvider",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "CcProtoLibraryProviders",
    srcs = ["CcProtoLibraryProviders.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)

java_library(
    name = "ProtoCcHeaderProvider",
    srcs = ["ProtoCcHeaderProvider.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)
