# External dependencies for the java_* rules.

new_local_repository(
    name = "local_jdk",
    path = DEFAULT_SYSTEM_JAVABASE,
    build_file = __embedded_dir__ + "/jdk.BUILD",
)

new_local_repository(
    name = "embedded_jdk",
    path = DEFAULT_SERVER_JAVABASE,
    build_file = __embedded_dir__ + "/jdk.BUILD",
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

# OpenJDK distributions that should only be downloaded on demand (e.g. when
# building a java_library or a genrule that uses java make variables).
# This will allow us to stop bundling the full JDK with Bazel.
# Note that while these are currently the same as the openjdk_* rules in
# Bazel's WORKSPACE file, but they don't have to be the same.
new_http_archive(
    name = "remotejdk_linux",
    sha256 = "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
    ],
    build_file = __embedded_dir__ + "/jdk.BUILD",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules",
)

new_http_archive(
    name = "remotejdk_macos",
    sha256 = "404e7058ff91f956612f47705efbee8e175a38b505fb1b52d8c1ea98718683de",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz",
    ],
    build_file = __embedded_dir__ + "/jdk.BUILD",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules",
)

new_http_archive(
    name = "remotejdk_win",
    sha256 = "e738829017f107e7a7cd5069db979398ec3c3f03ef56122f89ba38e7374f63ed",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip",
    ],
    build_file = __embedded_dir__ + "/jdk.BUILD",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-win_x64-allmodules",
)

# The source-code for this OpenJDK can be found at:
# https://openjdk.linaro.org/releases/jdk9-src-1708.tar.xz
new_http_archive(
    name = "remotejdk_linux_aarch64",
    sha256 = "72e7843902b0395e2d30e1e9ad2a5f05f36a4bc62529828bcbc698d54aec6022",
    urls = [
        # When you update this, also update the link to the source-code above.
        "http://openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
    ],
    build_file = __embedded_dir__ + "/jdk.BUILD",
    strip_prefix = "jdk9-server-release-1708",
)
