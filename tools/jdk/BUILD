package(default_visibility = ["//visibility:public"])

filegroup(
    name = "BUILD-jdk",
    srcs = [":BUILD.tools"],
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.java_tools",
        "BUILD.java_tools_prebuilt",
        "BUILD-jdk",  # Tools are build from the workspace for tests.
        "nosystemjdk/README",
        "proguard_whitelister.py",
        "proguard_whitelister_test.py",
        "proguard_whitelister_test_input.pgcfg",
        ":java_stub_template.txt",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
)

genrule(
    name = "copy_java_stub_template",
    srcs = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java:java_stub_template.txt",
    ],
    outs = ["java_stub_template.txt"],
    cmd = "cp $< $@",
    visibility = [
        "//tools:__pkg__",
        "//tools/jdk:__pkg__",
    ],
)
