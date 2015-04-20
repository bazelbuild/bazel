package(default_visibility = ["//visibility:public"])

filegroup(
    name = "jni_header",
    srcs = ["//external:jni_header"],
)

filegroup(
    name = "jni_md_header-darwin",
    srcs = ["//external:jni_md_header-darwin"],
)

filegroup(
    name = "jni_md_header-linux",
    srcs = ["//external:jni_md_header-linux"],
)

filegroup(
    name = "java",
    srcs = ["//external:java"],
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
    srcs = ["//external:bootclasspath"],
)

filegroup(
    name = "langtools",
    srcs = ["//external:langtools"],
)

java_import(
    name = "langtools-neverlink",
    jars = ["//external:langtools"],
    neverlink = 1,
)

# This one is just needed because of how filegroup redirection works.
filegroup(name = "jdk-null")

filegroup(
    name = "jdk",
    srcs = [
        ":jdk-null",
        "//external:jdk-default",
    ],
)

java_toolchain(
    name = "toolchain",
    encoding = "UTF-8",
    source_version = "8",
    target_version = "8",
)

filegroup(
    name = "JavaBuilder_deploy.jar",
    srcs = ["//external:__embedded_dir__/JavaBuilder_deploy.jar"],
)

filegroup(
    name = "SingleJar_deploy.jar",
    srcs = ["//external:__embedded_dir__/SingleJar_deploy.jar"],
)

filegroup(
    name = "ijar",
    srcs = ["//external:__embedded_dir__/ijar"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"],  # Tools are build from the workspace for tests.
)
