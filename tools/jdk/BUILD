package(default_visibility = ["//visibility:public"])

config_setting(
    name = "jdk7",
    values = {"define": "JAVA_VERSION=1.7"},
)

genrule(
    name = "BUILD-jdk7",
    srcs = [":BUILD"],
    outs = ["BUILD.jdk7"],
    cmd = "sed 's/_version = \"8\"/_version = \"7\"/' $< > $@",
)

filegroup(
    name = "BUILD-jdk",
    srcs = select({
        ":jdk7": [":BUILD-jdk7"],
        "//conditions:default": [":BUILD"],
    }),
)

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
    name = "jni_md_header-freebsd",
    srcs = ["//external:jni_md_header-freebsd"],
)

filegroup(
    name = "java",
    srcs = ["//external:java"],
)

filegroup(
    name = "jar",
    srcs = ["//external:jar"],
)

filegroup(
    name = "javac",
    srcs = ["//external:javac"],
)

# On Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent. Thus, we create a little filegroup that
# contains the appropriate platform-dependent file.
filegroup(
    name = "ijar",
    srcs = glob(["ijar/*"]),
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
    name = "extdir",
    srcs = ["//external:extdir"],
)

filegroup(
    name = "langtools",
    srcs = ["//third_party/java/jdk/langtools:javac_jar"],
)

java_import(
    name = "langtools-neverlink",
    jars = [":langtools"],
    neverlink = 1,
)

# This one is just needed because of how filegroup redirection works.
filegroup(name = "jre-null")

filegroup(
    name = "jre",
    srcs = [
        ":jre-null",
        "//external:jre-default",
    ],
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
    bootclasspath = [":bootclasspath"],
    encoding = "UTF-8",
    extclasspath = [":extdir"],
    genclass = ["//tools/jdk:GenClass_deploy.jar"],
    header_compiler = ["//tools/jdk:turbine_deploy.jar"],
    ijar = ["@bazel_tools//tools/jdk:ijar"],
    javabuilder = ["//tools/jdk:JavaBuilder_deploy.jar"],
    javac = ["//third_party/java/jdk/langtools:javac_jar"],
    jvm_opts = [
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
    ],
    singlejar = ["//tools/jdk:SingleJar_deploy.jar"],
    source_version = "8",
    target_version = "8",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD-jdk",  # Tools are build from the workspace for tests.
        "proguard_whitelister.py",
        "proguard_whitelister_test.py",
        "proguard_whitelister_test_input.cfg",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
)

py_binary(
    name = "proguard_whitelister",
    srcs = [
        "proguard_whitelister.py",
    ],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "proguard_whitelister_test",
    srcs = ["proguard_whitelister_test.py"],
    data = ["proguard_whitelister_test_input.cfg"],
    deps = [
        ":proguard_whitelister",
    ],
)
