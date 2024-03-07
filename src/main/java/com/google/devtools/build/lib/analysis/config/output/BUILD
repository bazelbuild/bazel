# Classes for generating configuration related output. For example, the output
# of the 'blaze config' or 'blaze cquery' commands.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "fragment_options_for_output",
    srcs = ["FragmentOptionsForOutput.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_options",
    ],
)

java_library(
    name = "configuration_for_output",
    srcs = ["ConfigurationForOutput.java"],
    deps = [
        ":fragment_for_output",
        ":fragment_options_for_output",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_class_set",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "fragment_for_output",
    srcs = ["FragmentForOutput.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_options",
    ],
)
