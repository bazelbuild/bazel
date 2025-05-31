load("@rules_fuzzing//fuzzing:java_defs.bzl", "java_fuzz_test")
load("@rules_java//java:defs.bzl", "java_binary", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/lib:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "RegexUtilTest",
    srcs = ["RegexUtilTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util/regex:regex_util",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_binary(
    name = "RegexUtilBenchmark",
    srcs = ["RegexUtilBenchmark.java"],
    main_class = "org.openjdk.jmh.Main",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util/regex:regex_util",
        "//third_party:jmh",
    ],
)

# Start fuzzing via
#   bazel run //src/test/java/com/google/devtools/build/lib/util/regex:RegexUtilFuzzTest_run
java_fuzz_test(
    name = "RegexUtilFuzzTest",
    srcs = ["RegexUtilFuzzTest.java"],
    jvm_flags = [
        "-Djazzer.instrumentation_includes=java.util.regex.*:com.google.devtools.build.lib.util.*",
        # The fuzz test intentionally passes untrusted input into Pattern#compile, don't report that fact.
        "-Djazzer.disabled_hooks=com.code_intelligence.jazzer.sanitizers.RegexInjection",
    ],
    tags = [
        "manual",
        "no_windows",
        "notap",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util/regex:regex_util",
        "//third_party:jmh",
    ],
)
