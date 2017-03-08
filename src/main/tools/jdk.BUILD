package(default_visibility = ["//visibility:public"])

filegroup(
    name = "jni_header",
    srcs = ["include/jni.h"],
)

filegroup(
    name = "jni_md_header-darwin",
    srcs = ["include/darwin/jni_md.h"],
)

filegroup(
    name = "jni_md_header-linux",
    srcs = ["include/linux/jni_md.h"],
)

filegroup(
    name = "jni_md_header-freebsd",
    srcs = ["include/freebsd/jni_md.h"],
)

filegroup(
    name = "jni_md_header-windows",
    srcs = ["include/win32/jni_md.h"],
)

filegroup(
    name = "java",
    srcs = select({
       ":windows" : ["bin/java.exe"],
       ":windows_msys" : ["bin/java.exe"],
       ":windows_msvc" : ["bin/java.exe"],
       "//conditions:default" : ["bin/java"],
    }),
)

filegroup(
    name = "jar",
    srcs = select({
       ":windows" : ["bin/jar.exe"],
       ":windows_msys" : ["bin/jar.exe"],
       ":windows_msvc" : ["bin/jar.exe"],
       "//conditions:default" : ["bin/jar"],
    }),
)

filegroup(
    name = "javac",
    srcs = select({
        ":windows" : ["bin/javac.exe"],
        ":windows_msys" : ["bin/javac.exe"],
        ":windows_msvc" : ["bin/javac.exe"],
        "//conditions:default" : ["bin/javac"],
    }),
)


filegroup(
    name = "xjc",
    srcs = ["bin/xjc"],
)

filegroup(
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

filegroup(
    name = "bootclasspath",
    srcs = ["jre/lib/%s" % jar for jar in BOOTCLASS_JARS],
)

# TODO(cushon): migrate to extclasspath and delete
filegroup(
    name = "extdir",
    srcs = glob(["jre/lib/ext/*.jar"]),
)

filegroup(
    name = "extclasspath",
    srcs = glob(["jre/lib/ext/*.jar"]),
)

filegroup(
    name = "jre-bin",
    srcs = select({
        # In some configurations, Java browser plugin is considered harmful and
        # common antivirus software blocks access to npjp2.dll interfering with Bazel,
        # so do not include it in JRE on Windows.
        ":windows" : glob(["jre/bin/**"], exclude = ["jre/bin/plugin2/**"]),
        ":windows_msys" : glob(["jre/bin/**"], exclude = ["jre/bin/plugin2/**"]),
        ":windows_msvc" : glob(["jre/bin/**"], exclude = ["jre/bin/plugin2/**"]),
        "//conditions:default" : glob(["jre/bin/**"])
    }),
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
)

filegroup(
    name = "jdk-bin",
    srcs = glob(
        ["bin/**"],
        # The JDK on Windows sometimes contains a directory called
        # "%systemroot%", which is not a valid label.
        exclude = ["**/*%*/**"]),
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
        ]),
)

java_runtime_suite(
    name = "jdk",
    runtimes = {},
    default = ":jdk-default",
)

java_runtime(
    name = "jdk-default",
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
)

java_import(
    name = "langtools-neverlink",
    jars = ["lib/tools.jar"],
    neverlink = 1,
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "windows_msys",
    values = {"cpu": "x64_windows_msys"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
    visibility = ["//visibility:private"],
)
