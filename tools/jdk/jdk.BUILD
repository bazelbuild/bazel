load("@rules_java//java:defs.bzl", "java_import", "java_runtime")

package(default_visibility = ["//visibility:public"])

exports_files(["BUILD.bazel"])

DEPRECATION_MESSAGE = ("Don't depend on targets in the JDK workspace;" +
                       " use @bazel_tools//tools/jdk:current_java_runtime instead" +
                       " (see https://github.com/bazelbuild/bazel/issues/5594)")

filegroup(
    name = "jni_header",
    srcs = ["include/jni.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jni_md_header-darwin",
    srcs = ["include/darwin/jni_md.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jni_md_header-linux",
    srcs = ["include/linux/jni_md.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jni_md_header-freebsd",
    srcs = ["include/freebsd/jni_md.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jni_md_header-openbsd",
    srcs = ["include/openbsd/jni_md.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jni_md_header-windows",
    srcs = ["include/win32/jni_md.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "java",
    srcs = select({
        ":windows": ["bin/java.exe"],
        "//conditions:default": ["bin/java"],
    }),
    data = [":jdk"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jar",
    srcs = select({
        ":windows": ["bin/jar.exe"],
        "//conditions:default": ["bin/jar"],
    }),
    data = [":jdk"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "javac",
    srcs = select({
        ":windows": ["bin/javac.exe"],
        "//conditions:default": ["bin/javac"],
    }),
    data = [":jdk"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "javadoc",
    srcs = select({
        ":windows": ["bin/javadoc.exe"],
        "//conditions:default": ["bin/javadoc"],
    }),
    data = [":jdk"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "xjc",
    srcs = ["bin/xjc"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "wsimport",
    srcs = ["bin/wsimport"],
    deprecation = DEPRECATION_MESSAGE,
)

BOOTCLASS_JARS = [
    "rt.jar",
    "resources.jar",
    "jsse.jar",
    "jce.jar",
    "charsets.jar",
]

# TODO(cushon): this isn't compatible with JDK 9
filegroup(
    name = "bootclasspath",
    srcs = ["jre/lib/%s" % jar for jar in BOOTCLASS_JARS],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jre-bin",
    srcs = select({
        # In some configurations, Java browser plugin is considered harmful and
        # common antivirus software blocks access to npjp2.dll interfering with Bazel,
        # so do not include it in JRE on Windows.
        ":windows": glob(
            ["jre/bin/**"],
            allow_empty = True,
            exclude = ["jre/bin/plugin2/**"],
        ),
        "//conditions:default": glob(
            ["jre/bin/**"],
            allow_empty = True,
        ),
    }),
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jre-lib",
    srcs = glob(
        ["jre/lib/**"],
        allow_empty = True,
    ),
)

filegroup(
    name = "jre",
    srcs = [":jre-default"],
)

filegroup(
    name = "jre-default",
    srcs = [
        ":jre-bin",
        ":jre-lib",
    ],
    deprecation = DEPRECATION_MESSAGE,
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

#This folder holds security policies
filegroup(
    name = "jdk-conf",
    srcs = glob(["conf/**"]),
)

filegroup(
    name = "jdk-include",
    srcs = glob(["include/**"]),
)

filegroup(
    name = "jdk-lib",
    srcs = glob(
        ["lib/**"],
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
        ":jre-default",
    ],
)

filegroup(
    name = "langtools",
    srcs = ["lib/tools.jar"],
    deprecation = DEPRECATION_MESSAGE,
)

java_import(
    name = "langtools-neverlink",
    deprecation = DEPRECATION_MESSAGE,
    jars = ["lib/tools.jar"],
    neverlink = 1,
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:private"],
)
