package(default_visibility = ["//visibility:public"])

filegroup(
    name = "jni_header",
    srcs = ["jdk/include/jni.h"],
)

filegroup(
    name = "jni_md_header-darwin",
    srcs = ["jdk/include/darwin/jni_md.h"],
)

filegroup(
    name = "jni_md_header-linux",
    srcs = ["jdk/include/linux/jni_md.h"],
)

filegroup(
    name = "java",
    srcs = ["jdk/bin/java"],
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
    srcs = ["jdk/jre/lib/%s" % jar for jar in BOOTCLASS_JARS],
)

filegroup(
    name = "langtools",
    srcs = ["jdk/lib/tools.jar"],
)

java_import(
    name = "langtools-neverlink",
    jars = ["jdk/lib/tools.jar"],
    neverlink = 1,
)

# This one is just needed because of how filegroup redirection works.
filegroup(name = "jdk-null")

filegroup(
    name = "jdk-default",
    srcs = glob(["jdk/bin/*"]),
    path = "jdk",
)

filegroup(
    name = "jdk",
    srcs = [
        ":jdk-default",
        ":jdk-null",
    ],
)

java_toolchain(
    name = "toolchain",
    encoding = "UTF-8",
    source_version = "8",
    target_version = "8",
)

exports_files([
    "JavaBuilder_deploy.jar",
    "ijar",
    "SingleJar_deploy.jar",
])
