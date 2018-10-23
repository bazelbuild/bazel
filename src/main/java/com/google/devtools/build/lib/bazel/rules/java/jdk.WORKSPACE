# External dependencies for the java_* rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

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
http_archive(
    name = "remotejdk_linux",
    sha256 = "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules",
)

http_archive(
    name = "remotejdk_macos",
    sha256 = "404e7058ff91f956612f47705efbee8e175a38b505fb1b52d8c1ea98718683de",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules",
)

http_archive(
    name = "remotejdk_win",
    sha256 = "e738829017f107e7a7cd5069db979398ec3c3f03ef56122f89ba38e7374f63ed",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-win_x64-allmodules",
)

# The source-code for this OpenJDK can be found at:
# https://openjdk.linaro.org/releases/jdk9-src-1708.tar.xz
http_archive(
    name = "remotejdk_linux_aarch64",
    sha256 = "72e7843902b0395e2d30e1e9ad2a5f05f36a4bc62529828bcbc698d54aec6022",
    urls = [
        # When you update this, also update the link to the source-code above.
        "https://mirror.bazel.build/openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
        "http://openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "jdk9-server-release-1708",
)

http_archive(
    name = "remotejdk10_linux",
    sha256 = "57fad3602e74c79587901d6966d3b54ef32cb811829a2552163185d5064fe9b5",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-linux_x64-allmodules.tar.gz",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "zulu10.2+3-jdk10.0.1-linux_x64-allmodules",
)


http_archive(
    name = "remotejdk10_macos",
    sha256 = "e669c9a897413d855b550b4e39d79614392e6fb96f494e8ef99a34297d9d85d3",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-macosx_x64-allmodules.tar.gz",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "zulu10.2+3-jdk10.0.1-macosx_x64-allmodules"
)

http_archive(
    name = "remotejdk10_win",
    sha256 = "c39e7700a8d41794d60985df5a20352435196e78ecbc6a2b30df7be8637bffd5",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-win_x64-allmodules.zip",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "zulu10.2+3-jdk10.0.1-win_x64-allmodules",
)

# The source-code for this OpenJDK can be found at:
# https://openjdk.linaro.org/releases/jdk10-src-1804.tar.xz
http_archive(
    name = "remotejdk10_linux_aarch64",
    sha256 = "b7098b7aaf6ee1ffd4a2d0371a0be26c5a5c87f6aebbe46fe9a92c90583a84be",
    urls = [
        # When you update this, also update the link to the source-code above.
        "https://mirror.bazel.build/openjdk.linaro.org/releases/jdk10-server-release-1804.tar.xz",
        "http://openjdk.linaro.org/releases/jdk10-server-release-1804.tar.xz",
    ],
    build_file = "@local_jdk//:BUILD.bazel",
    strip_prefix = "jdk10-server-release-1804",
)
