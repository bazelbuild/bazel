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
    name = "java",
    srcs = ["bin/java"],
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
    name = "jdk-default",
    srcs = glob(["bin/*"]),
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
