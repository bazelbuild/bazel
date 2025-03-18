load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "ComparingTransitionTest",
    srcs = ["ComparingTransitionTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:comparing_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ComposingTransitionFactoryTest",
    srcs = ["ComposingTransitionFactoryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:required_config_fragments_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:composing_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:split_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:transition_factory",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NoConfigTransitionTest",
    srcs = ["NoConfigTransitionTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_config_transition",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
