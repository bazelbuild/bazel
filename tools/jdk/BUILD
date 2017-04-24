package(default_visibility = ["//visibility:public"])

config_setting(
    name = "jdk7",
    values = {"define": "JAVA_VERSION=1.7"},
)

genrule(
    name = "BUILD-jdk7",
    srcs = [":BUILD"],
    outs = ["BUILD.jdk7"],
    cmd = "sed -e 's/_version = \"8\"/_version = \"7\"/' -e 's/javac_supports_workers = 1/javac_supports_workers = 0/' -e 's/forcibly_disable_header_compilation = 0/forcibly_disable_header_compilation = 1/g' -e 's/:JavaBuilder_deploy/:VanillaJavaBuilder_deploy/' $< > $@",
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
    actual = "@local_jdk//:jni_header",
)

alias(
    name = "jni_md_header-darwin",
    actual = "@local_jdk//:jni_md_header-darwin",
)

alias(
    name = "jni_md_header-linux",
    actual = "@local_jdk//:jni_md_header-linux",
)

alias(
    name = "jni_md_header-freebsd",
    actual = "@local_jdk//:jni_md_header-freebsd",
)

alias(
    name = "java",
    actual = "@local_jdk//:java",
)

alias(
    name = "jar",
    actual = "@local_jdk//:jar",
)

alias(
    name = "javac",
    actual = "@local_jdk//:javac",
)

# On Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent. Thus, we create a little filegroup that
# contains the appropriate platform-dependent file.
filegroup(
    name = "ijar",
    srcs = glob(["ijar/*"]),
)

filegroup(
    name = "genclass",
    srcs = ["//tools/jdk:GenClass_deploy.jar"],
)

filegroup(
    name = "turbine",
    srcs = ["//tools/jdk:turbine_deploy.jar"],
)

filegroup(
    name = "singlejar",
    srcs = ["//tools/jdk:SingleJar_deploy.jar"],
)

filegroup(
    name = "javabuilder",
    srcs = ["//tools/jdk:JavaBuilder_deploy.jar"],
)

filegroup(
    name = "vanillajavabuilder",
    srcs = ["//tools/jdk:VanillaJavaBuilder_deploy.jar"],
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
    actual = "@local_jdk//:bootclasspath",
)

alias(
    name = "extclasspath",
    actual = "@local_jdk//:extdir",
)

# TODO(cushon): migrate to extclasspath and delete
alias(
    name = "extdir",
    actual = "@local_jdk//:extdir",
)

filegroup(
    name = "langtools",
    srcs = ["//third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1"],
)

java_import(
    name = "langtools-neverlink",
    jars = [":langtools"],
    neverlink = 1,
)

alias(
    name = "jre",
    actual = "@local_jdk//:jre",
)

alias(
    name = "jdk",
    actual = "@local_jdk//:jdk",
)

java_toolchain(
    name = "toolchain",
    bootclasspath = [":bootclasspath"],
    encoding = "UTF-8",
    extclasspath = [":extclasspath"],
    forcibly_disable_header_compilation = 0,
    genclass = ["@bazel_tools//tools/jdk:genclass"],
    header_compiler = ["@bazel_tools//tools/jdk:turbine"],
    ijar = ["@bazel_tools//tools/jdk:ijar"],
    javabuilder = ["@bazel_tools//tools/jdk:javabuilder"],
    javac = ["//third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1"],
    javac_supports_workers = 1,
    jvm_opts = [
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
    ],
    misc = [
        "-XDskipDuplicateBridges=true",
    ],
    singlejar = ["@bazel_tools//tools/jdk:SingleJar_deploy.jar"],
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

java_import(
    name = "SingleJar",
    jars = ["SingleJar_deploy.jar"],
)

java_binary(
    name = "SingleJar_binary",
    main_class = "com.google.devtools.build.singlejar.SingleJar",
    runtime_deps = [":SingleJar"],
)
