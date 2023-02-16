package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "jsontypefactories",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:common",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:registry",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:jsr330_inject",
        "@maven//:com_google_code_gson_gson",
        "@maven//:com_google_guava_guava",
        "@maven//:com_ryanharter_auto_value_auto_value_gson_runtime",
    ],
)
