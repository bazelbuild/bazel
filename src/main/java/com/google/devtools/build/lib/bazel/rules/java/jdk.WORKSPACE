# External dependencies for the java_* rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/jdk:local_java_repository.bzl", "local_java_repository")

local_java_repository(
    name = "local_jdk",
    java_home = DEFAULT_SYSTEM_JAVABASE,
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
    name = "remotejdk15_linux",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    strip_prefix = "zulu15.27.17-ca-jdk15.0.0-linux_x64",
    sha256 = "0a38f1138c15a4f243b75eb82f8ef40855afcc402e3c2a6de97ce8235011b1ad",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu15.27.17-ca-jdk15.0.0-linux_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu15.27.17-ca-jdk15.0.0-linux_x64.tar.gz"
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk15_macos",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    strip_prefix = "zulu15.27.17-ca-jdk15.0.0-macosx_x64",
    sha256 = "f80b2e0512d9d8a92be24497334c974bfecc8c898fc215ce0e76594f00437482",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu15.27.17-ca-jdk15.0.0-macosx_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu15.27.17-ca-jdk15.0.0-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remotejdk15_win",
    build_file = "@bazel_tools//tools/jdk:jdk.BUILD",
    strip_prefix = "zulu15.27.17-ca-jdk15.0.0-win_x64",
    sha256 = "f535a530151e6c20de8a3078057e332b08887cb3ba1a4735717357e72765cad6",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu15.27.17-ca-jdk15.0.0-win_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu15.27.17-ca-jdk15.0.0-win_x64.zip"
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remote_java_tools_linux",
    sha256 = "7debb49457db2f10990f759b6984c6d4ccb4917a9a9fd402a6f224b5fb0e8e36",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v10.4/java_tools_javac11_linux-v10.4.zip",
        "https://github.com/bazelbuild/java_tools/releases/download/javac11_v10.4/java_tools_javac11_linux-v10.4.zip",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remote_java_tools_windows",
    sha256 = "3a7d802ccefffa6ecf6e190aaae081cf27fc0275e2a8dad7d3a943c63a4d0edc",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v10.4/java_tools_javac11_windows-v10.4.zip",
        "https://github.com/bazelbuild/java_tools/releases/download/javac11_v10.4/java_tools_javac11_windows-v10.4.zip",
    ],
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "remote_java_tools_darwin",
    sha256 = "587a7bc34c95a217a941b01a5a1f8ee48114fbe87e05872a41b7e019e5297a8a",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v10.4/java_tools_javac11_darwin-v10.4.zip",
        "https://github.com/bazelbuild/java_tools/releases/download/javac11_v10.4/java_tools_javac11_darwin-v10.4.zip",
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
    sha256 = "d0c573b94a6ef20ef6ff20154a23d0efcb409fb0e1ff0979cec318dfe42f0cdd",
    strip_prefix = "rules_cc-b1c40e1de81913a3c40e5948f78719c28152486d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        "https://github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
    ],
)

# Needed only because of java_tools.
maybe(
    http_archive,
    "rules_proto",
    sha256 = "8e7d59a5b12b233be5652e3d29f42fba01c7cbab09f6b3a8d0a57ed6d1e9a0da",
    strip_prefix = "rules_proto-7e4afce6fe62dbff0a4a03450143146f9f2d7488",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
    ],
)

register_toolchains("@bazel_tools//tools/jdk:all")
