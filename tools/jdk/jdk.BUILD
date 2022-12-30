load("@rules_java//java:defs.bzl", "java_runtime")

package(default_visibility = ["//visibility:public"])

exports_files(["BUILD.bazel"])

filegroup(
    name = "jre",
    srcs = glob(
        [
            "jre/bin/**",
            "jre/lib/**",
        ],
        allow_empty = True,
        # In some configurations, Java browser plugin is considered harmful and
        # common antivirus software blocks access to npjp2.dll interfering with Bazel,
        # so do not include it in JRE on Windows.
        exclude = ["jre/bin/plugin2/**"],
    ),
)

filegroup(
    name = "jdk-bin",
    srcs = glob(
        ["bin/**"],
        # The JDK on Windows sometimes contains a directory called
        # "%systemroot%", which is not a valid label.
        exclude = ["**/*%*/**"],
    ),
)

# This folder holds security policies.
filegroup(
    name = "jdk-conf",
    srcs = glob(
        ["conf/**"],
        allow_empty = True,
    ),
)

filegroup(
    name = "jdk-include",
    srcs = glob(
        ["include/**"],
        allow_empty = True,
    ),
)

filegroup(
    name = "jdk-lib",
    srcs = glob(
        ["lib/**", "release"],
        allow_empty = True,
        exclude = [
            "lib/missioncontrol/**",
            "lib/visualvm/**",
        ],
    ),
)

java_runtime(
    name = "jdk",
    srcs = [
        ":jdk-bin",
        ":jdk-conf",
        ":jdk-include",
        ":jdk-lib",
        ":jre",
    ],
)
