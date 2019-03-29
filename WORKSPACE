workspace(name = "io_bazel")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Protobuf expects an //external:python_headers label which would contain the
# Python headers if fast Python protos is enabled. Since we are not using fast
# Python protos, bind python_headers to a dummy target.
bind(
    name = "python_headers",
    actual = "//:dummy",
)

# Protobuf code generation for GRPC requires three external labels:
# //external:grpc-java_plugin
# //external:grpc-jar
# //external:guava
bind(
    name = "grpc-java-plugin",
    actual = "//third_party/grpc:grpc-java-plugin",
)

bind(
    name = "grpc-jar",
    actual = "//third_party/grpc:grpc-jar",
)

bind(
    name = "guava",
    actual = "//third_party:guava",
)

# Used by //third_party/protobuf:protobuf_python
bind(
    name = "six",
    actual = "//third_party/py/six",
)

http_archive(
    name = "bazel_j2objc",
    # Computed using "shasum -a 256 j2objc-2.0.3.zip"
    sha256 = "a36bac432d0dbd8c98249e484b2b69dd5720afa4abb58711a3c3def1c0bfa21d",
    strip_prefix = "j2objc-2.0.3",
    urls = [
        "https://miirror.bazel.build/github.com/google/j2objc/releases/download/2.0.3/j2objc-2.0.3.zip",
        "https://github.com/google/j2objc/releases/download/2.0.3/j2objc-2.0.3.zip",
    ],
)

# For src/test/shell/bazel:test_srcs
load("//src/test/shell/bazel:list_source_repository.bzl", "list_source_repository")

list_source_repository(name = "local_bazel_source_list")

# To run the Android integration tests in //src/test/shell/bazel/android:all or
# build the Android sample app in //examples/android/java/bazel:hello_world
#
#   1. Install an Android SDK and NDK from https://developer.android.com
#   2. Set the $ANDROID_HOME and $ANDROID_NDK_HOME environment variables
#   3. Uncomment the two lines below
#
# android_sdk_repository(name = "androidsdk")
# android_ndk_repository(name = "androidndk")

# In order to run //src/test/shell/bazel:maven_skylark_test, follow the
# instructions above for the Android integration tests and uncomment the
# following lines:
# load("//tools/build_defs/repo:maven_rules.bzl", "maven_dependency_plugin")
# maven_dependency_plugin()

# This allows rules written in skylark to locate apple build tools.
bind(
    name = "xcrunwrapper",
    actual = "@bazel_tools//tools/objc:xcrunwrapper",
)

new_local_repository(
    name = "com_google_protobuf",
    build_file = "./third_party/protobuf/3.6.1/BUILD",
    path = "./third_party/protobuf/3.6.1/",
)

local_repository(
    name = "googleapis",
    path = "./third_party/googleapis/",
)

local_repository(
    name = "remoteapis",
    path = "./third_party/remoteapis/",
)

http_archive(
    name = "desugar_jdk_libs",
    # Commit e0b0291b2c51fbe5a7cfa14473a1ae850f94f021 of 2018-12-4
    # Computed using "shasum -a 256 <zip>"
    sha256 = "fe2e04f91ce8c59d49d91b8102edc6627c6fa2906c1b0e7346f01419ec4f419d",
    strip_prefix = "desugar_jdk_libs-e0b0291b2c51fbe5a7cfa14473a1ae850f94f021",
    urls = [
        "https://mirror.bazel.build/github.com/google/desugar_jdk_libs/archive/e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip",
        "https://github.com/google/desugar_jdk_libs/archive/e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip",
    ],
)

load("//:distdir.bzl", "distdir_tar")

distdir_tar(
    name = "additional_distfiles",
    # Keep in sync with the archives fetched as part of building bazel
    archives = [
        "e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip",
        "f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz",
        "java_tools_pkg-0.5.1.tar.gz",
        "java_tools_javac10_linux-x86_64-v1.0.zip",
        "java_tools_javac10_windows-x86_64-v1.0_copy.zip",
        "java_tools_javac10_darwin-v1.0.zip",
        "2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz",
        "8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
        "0.16.2.zip",
    ],
    dirname = "derived/distdir",
    sha256 = {
        "e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip": "fe2e04f91ce8c59d49d91b8102edc6627c6fa2906c1b0e7346f01419ec4f419d",
        "f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz": "ba5d15ca230efca96320085d8e4d58da826d1f81b444ef8afccd8b23e0799b52",
        "java_tools_pkg-0.5.1.tar.gz": "bcfc1a3dd0d638a49fed50f17f0f1f7d77101debf19ae2c82119c82459a9a8d1",
        "java_tools_javac10_linux-x86_64-v1.0.zip": "3eb952a53de09163024ad6c1ee964d6261bd6af2b1a644ea930d1b49f1cdfcc6",
        "java_tools_javac10_windows-x86_64-v1.0_copy.zip": "56adebe4a120dea79c7cc1c0249490d3fcbadb1874d8b58020ab032d99b7fcf4",
        "java_tools_javac10_darwin-v1.0.zip": "f5cd64d0474dca3ae4cef8e0325d7f5955223bff95dade15ce797821512bcd12",
        "2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz": "4a1318fed4831697b83ce879b3ab70ae09592b167e5bda8edaff45132d1c3b3f",
        "8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz": "d868ce50d592ef4aad7dec4dd32ae68d2151261913450fac8390b3fd474bb898",
        "0.16.2.zip": "9b72bb0aea72d7cbcfc82a01b1e25bf3d85f791e790ddec16c65e2d906382ee0",
    },
    urls = {
        "e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip": [
            "https://mirror.bazel.build/github.com/google/desugar_jdk_libs/archive/e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip",
            "https://github.com/google/desugar_jdk_libs/archive/e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip",
        ],
        "f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/archive/f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/archive/f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz",
        ],
        "java_tools_pkg-0.5.1.tar.gz": [
            "https://mirror.bazel.build/bazel_java_tools/java_tools_pkg-0.5.1.tar.gz",
        ],
        "java_tools_javac10_linux-x86_64-v1.0.zip": [
            "https://mirror.bazel.build/bazel_java_tools/java_tools_javac10_linux-x86_64-v1.0.zip",
        ],
        "java_tools_javac10_windows-x86_64-v1.0_copy.zip": [
            "https://mirror.bazel.build/bazel_java_tools/java_tools_javac10_windows-x86_64-v1.0_copy.zip",
        ],
        "java_tools_javac10_darwin-v1.0.zip": [
            "https://mirror.bazel.build/bazel_java_tools/java_tools_javac10_darwin-v1.0.zip",
        ],
        "2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/skydoc/archive/2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz",
            "https://github.com/bazelbuild/skydoc/archive/2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz",
        ],
        "8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_sass/archive/8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
            "https://github.com/bazelbuild/rules_sass/archive/8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
        ],
        "0.16.2.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/archive/0.16.2.zip",
            "https://github.com/bazelbuild/rules_nodejs/archive/0.16.2.zip",
        ],
    },
)

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
http_file(
    name = "openjdk_linux",
    downloaded_file_path = "zulu-linux.tar.gz",
    sha256 = "460d8a4f0c0204160b48086e341b22943c9cca471b195340e75b38ae9eb33c1c",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-linux_x64-allmodules-90755145cb6e6418584d8603cd5fa9afbb30aecc-1549209950.tar.gz",
    ],
)

http_file(
    name = "openjdk_linux_vanilla",
    downloaded_file_path = "zulu-linux-vanilla.tar.gz",
    sha256 = "f3f44b6235508e87b760bf37a49e186cc1fa4e9cd28384c4dbf5a33991921e08",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz",
    ],
)

http_file(
    name = "openjdk_linux_minimal",
    downloaded_file_path = "zulu-linux-minimal.tar.gz",
    sha256 = "734cbfc9a5264e6cff4877d6694acb2305359531a503748879a84d4996b24932",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-linux_x64-minimal-7a3af9f6f98ce69c1ebd2931817c2664a18cf279-1552657479.tar.gz",
    ],
)

# Used by CI to test Bazel on platforms without an installed system JDK.
# TODO(twerth): Migrate to @remotejdk when https://github.com/bazelbuild/bazel/pull/6216 is merged.
http_archive(
    name = "openjdk_linux_archive",
    build_file_content = "java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])",
    sha256 = "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos",
    downloaded_file_path = "zulu-macos.tar.gz",
    sha256 = "8fa61d85ca6f657d646fdb50cfc8634987f8f7d8a3250ed39fb7364647633252",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-macosx_x64-allmodules-90755145cb6e6418584d8603cd5fa9afbb30aecc-1549209951.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos_vanilla",
    downloaded_file_path = "zulu-macos-vanilla.tar.gz",
    sha256 = "059f8e3484bf07b63a8f2820d5f528f473eff1befdb1896ee4f8ff06be3b8d8f",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip",
    ],
)

http_file(
    name = "openjdk_macos_minimal",
    downloaded_file_path = "zulu-macos-minimal.tar.gz",
    sha256 = "fd2a39d8947c53879f3d809c2ff783fc279d96f4ef2cccefb34d1d1b9cc6165f",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-macosx_x64-minimal-7a3af9f6f98ce69c1ebd2931817c2664a18cf279-1552657467.tar.gz",
    ],
)

http_file(
    name = "openjdk_win",
    downloaded_file_path = "zulu-win.zip",
    sha256 = "e6ddb361309f8e84eb5fb5ad8b0f5cc031ba3679910139262c31efd8f7579d05",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-win_x64-allmodules-90755145cb6e6418584d8603cd5fa9afbb30aecc-1549209972.zip",
    ],
)

http_file(
    name = "openjdk_win_vanilla",
    downloaded_file_path = "zulu-win-vanilla.zip",
    sha256 = "e1f5b4ce1b9148140fae2fcfb8a96d1c9b7eac5b8df0e13fbcad9b8561284880",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-win_x64.zip",
    ],
)

http_file(
    name = "openjdk_win_minimal",
    downloaded_file_path = "zulu-win-minimal.zip",
    sha256 = "ce063face0ca9d539084731edf2ec92a3faa3c1d1b6915cf5ec1a09facbf52ac",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-win_x64-minimal-7a3af9f6f98ce69c1ebd2931817c2664a18cf279-1552657495.zip",
    ],
)

# The source-code for this OpenJDK can be found at:
# https://openjdk.linaro.org/releases/jdk9-src-1708.tar.xz
http_file(
    name = "openjdk_linux_aarch64",
    sha256 = "72e7843902b0395e2d30e1e9ad2a5f05f36a4bc62529828bcbc698d54aec6022",
    urls = [
        # When you update this, also update the link to the source-code above.
        "https://mirror.bazel.build/openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
        "http://openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
    ],
)

http_archive(
    name = "bazel_toolchains",
    sha256 = "4b1468b254a572dbe134cc1fd7c6eab1618a72acd339749ea343bd8f55c3b7eb",
    strip_prefix = "bazel-toolchains-d665ccfa3e9c90fa789671bf4ef5f7c19c5715c4",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/d665ccfa3e9c90fa789671bf4ef5f7c19c5715c4.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/d665ccfa3e9c90fa789671bf4ef5f7c19c5715c4.tar.gz",
    ],
)

# Creates toolchain configuration for remote execution with BuildKite CI
# for rbe_ubuntu1604.
# To run the tests with RBE on BuildKite CI uncomment the two lines below
# load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")
# rbe_autoconfig(name = "buildkite_config")

# We're pinning to a commit because this project does not have a recent release.
# Nothing special about this commit, though.
http_archive(
    name = "com_google_googletest",
    sha256 = "0fb00ff413f6b9b80ccee44a374ca7a18af7315aea72a43c62f2acd1ca74e9b5",
    strip_prefix = "googletest-f13bbe2992d188e834339abe6f715b2b2f840a77",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/f13bbe2992d188e834339abe6f715b2b2f840a77.tar.gz",
        "https://github.com/google/googletest/archive/f13bbe2992d188e834339abe6f715b2b2f840a77.tar.gz",
    ],
)

http_archive(
    name = "bazel_skylib",
    # Commit f83cb8dd6f5658bc574ccd873e25197055265d1c of 2018-11-26
    sha256 = "ba5d15ca230efca96320085d8e4d58da826d1f81b444ef8afccd8b23e0799b52",
    strip_prefix = "bazel-skylib-f83cb8dd6f5658bc574ccd873e25197055265d1c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/archive/f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/archive/f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz",
    ],
)

# Note that skydoc depends on being called io_bazel_skydoc (and not just skydoc)
# to work without being patched, as it hard-codes this name in its sources.
http_archive(
    name = "io_bazel_skydoc",
    sha256 = "4a1318fed4831697b83ce879b3ab70ae09592b167e5bda8edaff45132d1c3b3f",
    strip_prefix = "skydoc-2d9566b21fbe405acf5f7bf77eda30df72a4744c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/skydoc/archive/2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz",
        "https://github.com/bazelbuild/skydoc/archive/2d9566b21fbe405acf5f7bf77eda30df72a4744c.tar.gz",
    ],
)

# For testing, have an distdir_tar with all the archives implicit in every
# WORKSPACE, to that they don't have to be refetched for every test
# calling `bazel sync`.
distdir_tar(
    name = "test_WORKSPACE_files",
    archives = [
        "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
        "zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz",
        "zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip",
        "jdk9-server-release-1708.tar.xz",
        "zulu10.2+3-jdk10.0.1-linux_x64-allmodules.tar.gz",
        "zulu10.2+3-jdk10.0.1-macosx_x64-allmodules.tar.gz",
        "zulu10.2+3-jdk10.0.1-win_x64-allmodules.zip",
        "jdk10-server-release-1804.tar.xz",
        "java_tools_javac10_linux-x86_64-v1.0.zip",
        "java_tools_javac10_windows-x86_64-v1.0_copy.zip",
        "java_tools_javac10_darwin-v1.0.zip",
        "zulu11.2.3-jdk11.0.1-linux_x64.tar.gz",
        "zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz",
        "zulu11.2.3-jdk11.0.1-win_x64.zip",
        "zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz",
        "zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip",
        "zulu11.29.3-ca-jdk11.0.2-win_x64.zip",
        "1a6aaad88aaaf186c168623c872c55ed80c05c5a.zip",  # TODO(jin): do a proper release
    ],
    dirname = "test_WORKSPACE/distdir",
    sha256 = {
        "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz": "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
        "zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz": "404e7058ff91f956612f47705efbee8e175a38b505fb1b52d8c1ea98718683de",
        "zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip": "e738829017f107e7a7cd5069db979398ec3c3f03ef56122f89ba38e7374f63ed",
        "jdk9-server-release-1708.tar.xz": "72e7843902b0395e2d30e1e9ad2a5f05f36a4bc62529828bcbc698d54aec6022",
        "zulu10.2+3-jdk10.0.1-linux_x64-allmodules.tar.gz": "57fad3602e74c79587901d6966d3b54ef32cb811829a2552163185d5064fe9b5",
        "zulu10.2+3-jdk10.0.1-macosx_x64-allmodules.tar.gz": "e669c9a897413d855b550b4e39d79614392e6fb96f494e8ef99a34297d9d85d3",
        "zulu10.2+3-jdk10.0.1-win_x64-allmodules.zip": "c39e7700a8d41794d60985df5a20352435196e78ecbc6a2b30df7be8637bffd5",
        "jdk10-server-release-1804.tar.xz": "b7098b7aaf6ee1ffd4a2d0371a0be26c5a5c87f6aebbe46fe9a92c90583a84be",
        "java_tools_javac10_linux-x86_64-v1.0.zip": "3eb952a53de09163024ad6c1ee964d6261bd6af2b1a644ea930d1b49f1cdfcc6",
        "java_tools_javac10_windows-x86_64-v1.0_copy.zip": "56adebe4a120dea79c7cc1c0249490d3fcbadb1874d8b58020ab032d99b7fcf4",
        "java_tools_javac10_darwin-v1.0.zip": "f5cd64d0474dca3ae4cef8e0325d7f5955223bff95dade15ce797821512bcd12",
        "zulu11.2.3-jdk11.0.1-linux_x64.tar.gz": "232b1c3511f0d26e92582b7c3cc363be7ac633e371854ca2f2e9f2b50eb72a75",
        "zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz": "1edf366ee821e5db8e348152fcb337b28dfd6bf0f97943c270dcc6747cedb6cb",
        "zulu11.2.3-jdk11.0.1-win_x64.zip": "8e1e2b8347de6746f3fd1538840dd643201533ab113abc4ed93678e342d28aa3",
        "zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz": "f3f44b6235508e87b760bf37a49e186cc1fa4e9cd28384c4dbf5a33991921e08",
        "zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip": "059f8e3484bf07b63a8f2820d5f528f473eff1befdb1896ee4f8ff06be3b8d8f",
        "zulu11.29.3-ca-jdk11.0.2-win_x64.zip": "e1f5b4ce1b9148140fae2fcfb8a96d1c9b7eac5b8df0e13fbcad9b8561284880",
        "1a6aaad88aaaf186c168623c872c55ed80c05c5a.zip": "7210c2c5dcd2053165c68f00ae83c37ac4a6b04cd309193ded12fe828d875ed9",
    },
    urls = {
        "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz"],
        "zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz"],
        "zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip": ["https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip"],
        "jdk9-server-release-1708.tar.xz": ["https://mirror.bazel.build/openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz"],
        "zulu10.2+3-jdk10.0.1-linux_x64-allmodules.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-linux_x64-allmodules.tar.gz"],
        "zulu10.2+3-jdk10.0.1-macosx_x64-allmodules.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-macosx_x64-allmodules.tar.gz"],
        "zulu10.2+3-jdk10.0.1-win_x64-allmodules.zip": ["https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-win_x64-allmodules.zip"],
        "jdk10-server-release-1804.tar.xz": ["https://mirror.bazel.build/openjdk.linaro.org/releases/jdk10-server-release-1804.tar.xz"],
        "java_tools_javac10_linux-x86_64-v1.0.zip": ["https://mirror.bazel.build/bazel_java_tools/java_tools_javac10_linux-x86_64-v1.0.zip"],
        "java_tools_javac10_windows-x86_64-v1.0_copy.zip": ["https://mirror.bazel.build/bazel_java_tools/java_tools_javac10_windows-x86_64-v1.0_copy.zip"],
        "java_tools_javac10_darwin-v1.0.zip": ["https://mirror.bazel.build/bazel_java_tools/java_tools_javac10_darwin-v1.0.zip"],
        "zulu11.2.3-jdk11.0.1-linux_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-linux_x64.tar.gz"],
        "zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz"],
        "zulu11.2.3-jdk11.0.1-win_x64.zip": ["https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-win_x64.zip"],
        "zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz"],
        "zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip": ["https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip"],
        "zulu11.29.3-ca-jdk11.0.2-win_x64.zip": ["https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-win_x64.zip"],
        "1a6aaad88aaaf186c168623c872c55ed80c05c5a.zip": ["https://github.com/jin/legacy_android_tools/archive/1a6aaad88aaaf186c168623c872c55ed80c05c5a.zip"],
    },
)

load("//scripts/docs:doc_versions.bzl", "DOC_VERSIONS")

[http_file(
    name = "jekyll_tree_%s" % DOC_VERSION["version"].replace(".", "_"),
    sha256 = DOC_VERSION["sha256"],
    urls = ["https://mirror.bazel.build/bazel_versioned_docs/jekyll-tree-%s.tar" % DOC_VERSION["version"]],
) for DOC_VERSION in DOC_VERSIONS]

# Skydoc recommends declaring its dependencies via "*_dependencies" functions.
# This requires that the repositories these functions come from need to be
# fetched unconditionally for everything (including just building bazel!), so
# provide them as http_archives that can be shiped in the distdir, to keep the
# distribution archive self-contained.
http_archive(
    name = "io_bazel_rules_sass",
    sha256 = "d868ce50d592ef4aad7dec4dd32ae68d2151261913450fac8390b3fd474bb898",
    strip_prefix = "rules_sass-8ccf4f1c351928b55d5dddf3672e3667f6978d60",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_sass/archive/8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
        "https://github.com/bazelbuild/rules_sass/archive/8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
    ],
)

http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "9b72bb0aea72d7cbcfc82a01b1e25bf3d85f791e790ddec16c65e2d906382ee0",
    strip_prefix = "rules_nodejs-0.16.2",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/archive/0.16.2.zip",
        "https://github.com/bazelbuild/rules_nodejs/archive/0.16.2.zip",
    ],
)

load("@io_bazel_skydoc//:setup.bzl", "skydoc_repositories")

skydoc_repositories()

load("@io_bazel_rules_sass//:package.bzl", "rules_sass_dependencies")

rules_sass_dependencies()

load("@build_bazel_rules_nodejs//:defs.bzl", "node_repositories")

node_repositories()

load("@io_bazel_rules_sass//:defs.bzl", "sass_repositories")

sass_repositories()
