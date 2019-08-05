# External dependencies for the java_* rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

new_local_repository(
    name = "local_jdk",
    build_file = __embedded_dir__ + "/jdk.BUILD",
    path = DEFAULT_SYSTEM_JAVABASE,
)

bind(
    name = "bootclasspath",
    actual = "@local_jdk//:bootclasspath",
)

# TODO(cushon): migrate to extclasspath and delete
bind(
    name = "extdir",
    actual = "@local_jdk//:extdir",
)

bind(
    name = "extclasspath",
    actual = "@local_jdk//:extdir",
)

bind(
    name = "jni_header",
    actual = "@local_jdk//:jni_header",
)

bind(
    name = "jni_md_header-darwin",
    actual = "@local_jdk//:jni_md_header-darwin",
)

bind(
    name = "jni_md_header-linux",
    actual = "@local_jdk//:jni_md_header-linux",
)

bind(
    name = "jni_md_header-freebsd",
    actual = "@local_jdk//:jni_md_header-freebsd",
)

bind(
    name = "java",
    actual = "@local_jdk//:java",
)

bind(
    name = "jar",
    actual = "@local_jdk//:jar",
)

bind(
    name = "javac",
    actual = "@local_jdk//:javac",
)

bind(
    name = "jre",
    actual = "@local_jdk//:jre",
)

bind(
    name = "jdk",
    actual = "@local_jdk//:jdk",
)

# TODO: Remove these two rules after we've migrated. In order to properly look
# up Jdks/Jres for cross-platform builds, the lookup needs to happen in the Jdk
# repository. For now, use an alias rule that redirects to //external:{jre,jdk}.
bind(
    name = "jre-default",
    actual = "@local_jdk//:jre",
)

bind(
    name = "jdk-default",
    actual = "@local_jdk//:jdk",
)

maybe(
    http_archive,
    "rules_java",
    sha256 = "af7eb0fcbc518f88e66d239de428fc87e919514c96ab09589bfee756381db297",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/de5b585b1367463ac42b44db85447189301dc5e1.zip",
        "https://github.com/bazelbuild/rules_java/archive/de5b585b1367463ac42b44db85447189301dc5e1.zip",
    ],
    strip_prefix = "rules_java-de5b585b1367463ac42b44db85447189301dc5e1",
)

# Needed only because of java_tools.
maybe(
    http_archive,
    "rules_cc",
    sha256 = "36fa66d4d49debd71d05fba55c1353b522e8caef4a20f8080a3d17cdda001d89",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/0d5f3f2768c6ca2faca0079a997a97ce22997a0c.zip",
        "https://github.com/bazelbuild/rules_cc/archive/0d5f3f2768c6ca2faca0079a997a97ce22997a0c.zip",
    ],
    strip_prefix = "rules_cc-0d5f3f2768c6ca2faca0079a997a97ce22997a0c",
)

register_toolchains("@bazel_tools//tools/jdk:all")
