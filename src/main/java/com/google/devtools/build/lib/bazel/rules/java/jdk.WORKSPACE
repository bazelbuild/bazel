# External dependencies for the java_* rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/jdk:local_java_repository.bzl", "local_java_repository")

local_java_repository(
    name = "local_jdk",
    java_home = DEFAULT_SYSTEM_JAVABASE,
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
    name = "jni_md_header-openbsd",
    actual = "@local_jdk//:jni_md_header-openbsd",
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

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk11_linux",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "360626cc19063bc411bfed2914301b908a8f77a7919aaea007a977fa8fb3cde1",
    strip_prefix = "zulu11.37.17-ca-jdk11.0.6-linux_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk11_linux_aarch64",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "a452f1b9682d9f83c1c14e54d1446e1c51b5173a3a05dcb013d380f9508562e4",
    strip_prefix = "zulu11.37.48-ca-jdk11.0.6-linux_aarch64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.37.48-ca-jdk11.0.6/zulu11.37.48-ca-jdk11.0.6-linux_aarch64.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk11_linux_ppc64le",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "a417db0295b1f4b538ecbaf7c774f3a177fab9657a665940170936c0eca4e71a",
    strip_prefix = "jdk-11.0.7+10",
    urls = [
        "https://mirror.bazel.build/openjdk/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.7_10.tar.gz",
        "https://github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.7_10.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk11_linux_s390x",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "d9b72e87a1d3ebc0c9552f72ae5eb150fffc0298a7cb841f1ce7bfc70dcd1059",
    strip_prefix = "jdk-11.0.7+10",
    urls = [
        "https://mirror.bazel.build/github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.7_10.tar.gz",
        "https://github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.7_10.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk11_macos",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "e1fe56769f32e2aaac95e0a8f86b5a323da5af3a3b4bba73f3086391a6cc056f",
    strip_prefix = "zulu11.37.17-ca-jdk11.0.6-macosx_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk11_win",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "a9695617b8374bfa171f166951214965b1d1d08f43218db9a2a780b71c665c18",
    strip_prefix = "zulu11.37.17-ca-jdk11.0.6-win_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-win_x64.zip",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk14_linux",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "48bb8947034cd079ad1ef83335e7634db4b12a26743a0dc314b6b861480777aa",
    strip_prefix = "zulu14.28.21-ca-jdk14.0.1-linux_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu14.28.21-ca-jdk14.0.1-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk14_macos",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "088bd4d0890acc9f032b738283bf0f26b2a55c50b02d1c8a12c451d8ddf080dd",
    strip_prefix = "zulu14.28.21-ca-jdk14.0.1-macosx_x64",
    urls = ["https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu14.28.21-ca-jdk14.0.1-macosx_x64.tar.gz"],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk14_win",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    sha256 = "9cb078b5026a900d61239c866161f0d9558ec759aa15c5b4c7e905370e868284",
    strip_prefix = "zulu14.28.21-ca-jdk14.0.1-win_x64",
    urls = ["https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu14.28.21-ca-jdk14.0.1-win_x64.zip"],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remote_java_tools_linux",
    sha256 = "69e65353c2cd65780abcbcce4daae973599298273b0f8b4d469eed822cb220d1",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/release_candidates/javac11/v10.0/java_tools_javac11_linux-v10.0-rc1.zip",
        "https://github.com/bazelbuild/java_tools/release_candidates/download/javac11_v10.0/java_tools_javac11_linux-v10.0-rc1.zip",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remote_java_tools_windows",
    sha256 = "d2f62af8daa0a3d55789b605f6582e37038329c64843337c71e64515468e55c4",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/release_candidates/javac11/v10.0/java_tools_javac11_windows-v10.0-rc1.zip",
        "https://github.com/bazelbuild/java_tools/release_candidates/download/javac11_v10.0/java_tools_javac11_windows-v10.0-rc1.zip",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remote_java_tools_darwin",
    sha256 = "64e5de2175dfccb96831573946b80d106edf3801d9db38b564514bf3581d466b",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/release_candidates/javac11/v10.0/java_tools_javac11_darwin-v10.0-rc1.zip",
        "https://github.com/bazelbuild/java_tools/release_candidates/download/javac11_v10.0/java_tools_javac11_darwin-v10.0-rc1.zip",
    ],
)

maybe(
    http_archive,
    "rules_java",
    sha256 = "bc81f1ba47ef5cc68ad32225c3d0e70b8c6f6077663835438da8d5733f917598",
    strip_prefix = "rules_java-7cf3cefd652008d0a64a419c34c13bdca6c8f178",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        "https://github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
    ],
)

# Needed only because of java_tools.
maybe(
    http_archive,
    "rules_cc",
    sha256 = "1d4dbbd1e1e9b57d40bb0ade51c9e882da7658d5bfbf22bbd15b68e7879d761f",
    strip_prefix = "rules_cc-8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
        "https://github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
    ],
)

# Needed only because of java_tools.
maybe(
    http_archive,
    "rules_proto",
    sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
    strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
    ],
)

register_toolchains("@bazel_tools//tools/jdk:all")
