package(default_visibility = ["//visibility:public"])

config_setting(
    name = "jdk7",
    values = {"define": "JAVA_VERSION=1.7"},
)

genrule(
    name = "BUILD-jdk7",
    srcs = [":BUILD"],
    outs = ["BUILD.jdk7"],
    cmd = "sed -e 's/_version = \"8\"/_version = \"7\"/' -e 's/javac_supports_workers = 1/javac_supports_workers = 0/' $< > $@",
)

filegroup(
    name = "BUILD-jdk",
    srcs = select({
        ":jdk7": [":BUILD-jdk7"],
        "//conditions:default": [":BUILD"],
    }),
)

alias(
    name = "jni_header",
    actual = "//external:jni_header",
)

alias(
    name = "jni_md_header-darwin",
    actual = "//external:jni_md_header-darwin",
)

alias(
    name = "jni_md_header-linux",
    actual = "//external:jni_md_header-linux",
)

alias(
    name = "jni_md_header-freebsd",
    actual = "//external:jni_md_header-freebsd",
)

alias(
    name = "java",
    actual = "//external:java",
)

alias(
    name = "jar",
    actual = "//external:jar",
)

alias(
    name = "javac",
    actual = "//external:javac",
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

alias(
    name = "bootclasspath",
    actual = "//external:bootclasspath",
)

alias(
    name = "extdir",
    actual = "//external:extdir",
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

alias(
    name = "jre",
    actual = "//external:jre",
)

alias(
    name = "jdk",
    actual = "//external:jdk",
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
    javac_supports_workers = 1,
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

# For java coverage
alias(
    name = "jacoco-blaze-agent",
    actual = "//third_party/java/jacoco:blaze-agent",
)

java_import(
    name = "JacocoCoverage",
    jars = [":JacocoCoverage_deploy.jar"],
)
