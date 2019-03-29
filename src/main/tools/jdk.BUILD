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
    name = "jni_md_header-windows",
    srcs = ["include/win32/jni_md.h"],
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "java",
    deprecation = DEPRECATION_MESSAGE,
    srcs = select({
        ":windows": ["bin/java.exe"],
        "//conditions:default": ["bin/java"],
    }),
    data = [":jdk"],
)

filegroup(
    name = "jar",
    deprecation = DEPRECATION_MESSAGE,
    srcs = select({
        ":windows": ["bin/jar.exe"],
        "//conditions:default": ["bin/jar"],
    }),
    data = [":jdk"],
)

filegroup(
    deprecation = DEPRECATION_MESSAGE,
    name = "javac",
    srcs = select({
        ":windows": ["bin/javac.exe"],
        "//conditions:default": ["bin/javac"],
    }),
    data = [":jdk"],
)

filegroup(
    deprecation = DEPRECATION_MESSAGE,
    name = "javadoc",
    srcs = select({
        ":windows": ["bin/javadoc.exe"],
        "//conditions:default": ["bin/javadoc"],
    }),
    data = [":jdk"],
)

filegroup(
    deprecation = DEPRECATION_MESSAGE,
    name = "xjc",
    srcs = ["bin/xjc"],
)

filegroup(
    deprecation = DEPRECATION_MESSAGE,
    name = "wsimport",
    srcs = ["bin/wsimport"],
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
    deprecation = DEPRECATION_MESSAGE,
    name = "bootclasspath",
    srcs = ["jre/lib/%s" % jar for jar in BOOTCLASS_JARS],
)

# TODO(cushon): migrate to extclasspath and delete
filegroup(
    name = "extdir",
    deprecation = DEPRECATION_MESSAGE,
    srcs = glob(["jre/lib/ext/*.jar"]),
)

filegroup(
    name = "extclasspath",
    deprecation = DEPRECATION_MESSAGE,
    srcs = glob(["jre/lib/ext/*.jar"]),
)

filegroup(
    name = "jre-bin",
    srcs = select({
        # In some configurations, Java browser plugin is considered harmful and
        # common antivirus software blocks access to npjp2.dll interfering with Bazel,
        # so do not include it in JRE on Windows.
        ":windows": glob(["jre/bin/**"], exclude = ["jre/bin/plugin2/**"]),
        "//conditions:default": glob(["jre/bin/**"]),
    }),
    deprecation = DEPRECATION_MESSAGE,
)

filegroup(
    name = "jre-lib",
    srcs = glob(["jre/lib/**"]),
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
    jars = ["lib/tools.jar"],
    neverlink = 1,
    deprecation = DEPRECATION_MESSAGE,
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:private"],
)
