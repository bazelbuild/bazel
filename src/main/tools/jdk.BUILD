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
    name = "java",
    srcs = select({
       ":windows" : ["bin/java.exe"],
       "//conditions:default" : ["bin/java"],
    }),
)

filegroup(
    name = "jar",
    srcs = select({
       ":windows" : ["bin/jar.exe"],
       "//conditions:default" : ["bin/jar"],
    }),
)

filegroup(
    name = "javac",
    srcs = select({
        ":windows" : ["bin/javac.exe"],
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

filegroup(
    name = "extdir",
    srcs = glob(["jre/lib/ext/*.jar"]),
)

filegroup(
    name = "jre-bin",
    srcs = glob(["jre/bin/**"]),
)

filegroup(
    name = "jre-lib",
    srcs = glob(["jre/lib/**"]),
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
    srcs = glob(["bin/**"]),
)

filegroup(
    name = "jdk-include",
    srcs = glob(["include/**"]),
)

filegroup(
    name = "jdk-lib",
    srcs = glob(
        ["lib/**"],
        exclude= [
            "lib/missioncontrol/**",
            "lib/visualvm/**",
        ]),
)

filegroup(
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
