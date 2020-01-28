workspace(name = "io_bazel")

load("//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# These can be used as values for the patch_cmds and patch_cmds_win attributes
# of http_archive, in order to export the WORKSPACE file from the BUILD or
# BUILD.bazel file. This is useful for cases like //src:test_repos, where we
# have to be able to trigger a fetch of a repo by depending on it, but we don't
# actually want to build anything (so we can't depend on a target inside that
# repo).
EXPORT_WORKSPACE_IN_BUILD_FILE = [
    "test -f BUILD && chmod u+w BUILD || true",
    "echo >> BUILD",
    "echo 'exports_files([\"WORKSPACE\"], visibility = [\"//visibility:public\"])' >> BUILD",
]

EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE = [
    "test -f BUILD.bazel && chmod u+w BUILD.bazel || true",
    "echo >> BUILD.bazel",
    "echo 'exports_files([\"WORKSPACE\"], visibility = [\"//visibility:public\"])' >> BUILD.bazel",
]

EXPORT_WORKSPACE_IN_BUILD_FILE_WIN = [
    "Add-Content -Path BUILD -Value \"`nexports_files([`\"WORKSPACE`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
]

EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN = [
    "Add-Content -Path BUILD.bazel -Value \"`nexports_files([`\"WORKSPACE`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
]

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
    # Computed using "shasum -a 256 j2objc-2.5.zip"
    sha256 = "8d3403b5b7db57e347c943d214577f6879e5b175c2b59b7e075c0b6453330e9b",
    strip_prefix = "j2objc-2.5",
    urls = [
        "https://mirror.bazel.build/github.com/google/j2objc/releases/download/2.5/j2objc-2.5.zip",
        "https://github.com/google/j2objc/releases/download/2.5/j2objc-2.5.zip",
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

# This is a mock version of bazelbuild/rules_python that contains only
# @rules_python//python:defs.bzl. It is used by protobuf.
# TODO(#9029): We could potentially replace this with the real @rules_python.
new_local_repository(
    name = "rules_python",
    build_file = "//third_party/rules_python:BUILD",
    path = "./third_party/rules_python",
    workspace_file = "//third_party/rules_python:rules_python.WORKSPACE",
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
    # Keep in sync with the archives fetched as part of building bazel.
    archives = [
        "e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip",
        "f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz",
        "java_tools_javac11_linux-v7.0.zip",
        "java_tools_javac11_windows-v7.0.zip",
        "java_tools_javac11_darwin-v7.0.zip",
        "coverage_output_generator-v2.1.zip",
        "c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz",
        "8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
        "0.16.2.zip",
        "android_tools_pkg-0.14.tar.gz",
        # bazelbuild/platforms
        "46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
        # bazelbuild/rules_java
        "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        # bazelbuild/rules_cc
        "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
        # bazelbuild/bazel-toolchains
        "2.0.2.tar.gz",
        # bazelbuild/rules_pkg
        "rules_pkg-0.2.4.tar.gz",
        # bazelbuild/rules_proto
        "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
    ],
    dirname = "derived/distdir",
    sha256 = {
        "e0b0291b2c51fbe5a7cfa14473a1ae850f94f021.zip": "fe2e04f91ce8c59d49d91b8102edc6627c6fa2906c1b0e7346f01419ec4f419d",
        "f83cb8dd6f5658bc574ccd873e25197055265d1c.tar.gz": "ba5d15ca230efca96320085d8e4d58da826d1f81b444ef8afccd8b23e0799b52",
        "java_tools_javac11_linux-v7.0.zip": "3ff465e82954a70f49982610dd63f6f651beaa83c707dd637870b0e41cdcd2f0",
        "java_tools_javac11_windows-v7.0.zip": "11d90a147919e74d11870cdd58c4ee5de3062c08d11b16aa72d3f3bbfa9497a0",
        "java_tools_javac11_darwin-v7.0.zip": "373a4226906ae9ba908550da16e133c4cd1f01b8973af82b9a2eb6903cb4d645",
        "coverage_output_generator-v2.1.zip": "96ac6bc9b9fbc67b532bcae562da1642409791e6a4b8e522f04946ee5cc3ff8e",
        "c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz": "e6a76586b264f30679688f65f7e71ac112d1446681010a13bf22d9ca071f34b7",
        "8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz": "d868ce50d592ef4aad7dec4dd32ae68d2151261913450fac8390b3fd474bb898",
        "0.16.2.zip": "9b72bb0aea72d7cbcfc82a01b1e25bf3d85f791e790ddec16c65e2d906382ee0",
        "android_tools_pkg-0.14.tar.gz": "a3a951838448483e7af25afd10671b266cc6283104b4a2a427d31cac12cf0912",  # built at 6c63d70ef9c11a662b8323c0ae4f6d3ac53b1a60
        # bazelbuild/platforms
        "46993efdd33b73649796c5fc5c9efb193ae19d51.zip": "66184688debeeefcc2a16a2f80b03f514deac8346fe888fb7e691a52c023dd88",
        # bazelbuild/rules_java
        "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip": "bc81f1ba47ef5cc68ad32225c3d0e70b8c6f6077663835438da8d5733f917598",
        # bazelbuild/rules_cc
        "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip": "1d4dbbd1e1e9b57d40bb0ade51c9e882da7658d5bfbf22bbd15b68e7879d761f",
        # bazelbuild/bazel-toolchains
        "2.0.2.tar.gz": "a653c9d318e42b14c0ccd7ac50c4a2a276c0db1e39743ab88b5aa2f0bc9cf607",
        # bazelbuild/rules_pkg
        "rules_pkg-0.2.4.tar.gz": "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
        # bazelbuild/rules_proto
        "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz": "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
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
        "java_tools_javac11_linux-v7.0.zip": [
            "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_linux-v7.0.zip",
        ],
        "java_tools_javac11_windows-v7.0.zip": [
            "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_windows-v7.0.zip",
        ],
        "java_tools_javac11_darwin-v7.0.zip": [
            "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_darwin-v7.0.zip",
        ],
        "coverage_output_generator-v2.1.zip": [
            "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.1.zip",
        ],
        "c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/skydoc/archive/c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz",
            "https://github.com/bazelbuild/skydoc/archive/c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz",
        ],
        "8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_sass/archive/8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
            "https://github.com/bazelbuild/rules_sass/archive/8ccf4f1c351928b55d5dddf3672e3667f6978d60.tar.gz",
        ],
        "0.16.2.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/archive/0.16.2.zip",
            "https://github.com/bazelbuild/rules_nodejs/archive/0.16.2.zip",
        ],
        "android_tools_pkg-0.14.tar.gz": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.14.tar.gz",
        ],
        # bazelbuild/platforms
        "46993efdd33b73649796c5fc5c9efb193ae19d51.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/archive/46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
            "https://github.com/bazelbuild/platforms/archive/46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
        ],
        # bazelbuild/rules_java
        "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
            "https://github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        ],
        # bazelbuild/rules_cc
        "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
            "https://github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
        ],
        # bazelbuild/bazel-toolchains
        "2.0.2.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/2.0.2.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/releases/download/2.0.2/bazel-toolchains-2.0.2.tar.gz",
        ],
        # bazelbuild/rules_pkg
        "rules_pkg-0.2.4.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
        ],
        # bazelbuild/rules_proto
        "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
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
    sha256 = "5123bc8dd21886761d1fd9ca0fb1898b3372d7243064a070ec81ca9c9d1a6791",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-linux_x64-minimal-524ae2ca2a782c9f15e00f08bd35b3f8ceacbd7f-1556011926.tar.gz",
    ],
)

http_file(
    name = "openjdk_linux_aarch64",
    downloaded_file_path = "zulu-linux-aarch64.tar.gz",
    sha256 = "23c37c0c3a8fdcbc68e96e70ff5c5c020c14db76deaae9b547849afda4586e5e",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.15-ca-jdk11.0.3/zulu11.31.15-ca-jdk11.0.3-linux_aarch64-allmodules-c82eb4878c7dc829455caeb915affe36c89df06f-1561630858.tar.gz",
    ],
)

http_file(
    name = "openjdk_linux_aarch64_vanilla",
    downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
    sha256 = "3b0d91611b1bdc4d409afcf9eab4f0e7f4ae09f88fc01bd9f2b48954882ae69b",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.15-ca-jdk11.0.3/zulu11.31.15-ca-jdk11.0.3-linux_aarch64.tar.gz",
    ],
)

http_file(
    name = "openjdk_linux_aarch64_minimal",
    downloaded_file_path = "zulu-linux-aarch64-minimal.tar.gz",
    sha256 = "7af2583fe5ef0a781d4a9dca0c0160d42e7db1305ec1b66f98aa44c91cc875df",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.15-ca-jdk11.0.3/zulu11.31.15-ca-jdk11.0.3-linux_aarch64-minimal-c82eb4878c7dc829455caeb915affe36c89df06f-1561630858.tar.gz",
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
    sha256 = "ac56e44db46fd56ac78b39b6823daed4faa74a2677ac340c7d217f863884ec0f",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-macosx_x64-minimal-524ae2ca2a782c9f15e00f08bd35b3f8ceacbd7f-1556003114.tar.gz",
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
    sha256 = "8e5dada6e9ebcc9ce29b4d051449bb95d3ee1e620e166da862224bbf15211f8b",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-win_x64-minimal-524ae2ca2a782c9f15e00f08bd35b3f8ceacbd7f-1556003136.zip",
    ],
)

http_archive(
    name = "bazel_toolchains",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "a653c9d318e42b14c0ccd7ac50c4a2a276c0db1e39743ab88b5aa2f0bc9cf607",
    strip_prefix = "bazel-toolchains-2.0.2",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/2.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/2.0.2/bazel-toolchains-2.0.2.tar.gz",
    ],
)

load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")

rbe_autoconfig(
    name = "rbe_ubuntu1804_java11",
    detect_java_home = True,
    registry = "gcr.io",
    repository = "bazel-public/ubuntu1804/bazel",
    tag = "java11",
)

rbe_autoconfig(
    name = "rbe_ubuntu1604_java8",
    detect_java_home = True,
    registry = "gcr.io",
    repository = "bazel-public/ubuntu1604/bazel",
    tag = "java8",
)

# Creates toolchain configuration for remote execution with BuildKite CI
# for rbe_ubuntu1604.
# To run the tests with RBE on BuildKite CI uncomment the two lines below
# load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")
# rbe_autoconfig(name = "buildkite_config")

http_archive(
    name = "com_google_googletest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/release-1.10.0.tar.gz",
        "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
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
    sha256 = "e6a76586b264f30679688f65f7e71ac112d1446681010a13bf22d9ca071f34b7",
    strip_prefix = "skydoc-c7bbde2950769aac9a99364b0926230060a3ce04",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/skydoc/archive/c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz",
        "https://github.com/bazelbuild/skydoc/archive/c7bbde2950769aac9a99364b0926230060a3ce04.tar.gz",
    ],
)

http_archive(
    name = "rules_cc",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "1d4dbbd1e1e9b57d40bb0ade51c9e882da7658d5bfbf22bbd15b68e7879d761f",
    strip_prefix = "rules_cc-8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
        "https://github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
    ],
)

http_archive(
    name = "rules_java",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "bc81f1ba47ef5cc68ad32225c3d0e70b8c6f6077663835438da8d5733f917598",
    strip_prefix = "rules_java-7cf3cefd652008d0a64a419c34c13bdca6c8f178",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        "https://github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
    ],
)

http_archive(
    name = "rules_proto",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
    strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
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
        "java_tools_javac11_linux-v7.0.zip",
        "java_tools_javac11_windows-v7.0.zip",
        "java_tools_javac11_darwin-v7.0.zip",
        "coverage_output_generator-v2.1.zip",
        "zulu11.2.3-jdk11.0.1-linux_x64.tar.gz",
        "zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz",
        "zulu11.2.3-jdk11.0.1-win_x64.zip",
        "zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz",
        "zulu11.31.15-ca-jdk11.0.3-linux_aarch64.tar.gz",
        "zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip",
        "zulu11.29.3-ca-jdk11.0.2-win_x64.zip",
        "android_tools_pkg-0.14.tar.gz",
        # bazelbuild/platforms
        "46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
        # bazelbuild/rules_java
        "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        # bazelbuild/rules_cc
        "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
        # bazelbuild/rules_proto
        "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
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
        "java_tools_javac11_linux-v7.0.zip": "3ff465e82954a70f49982610dd63f6f651beaa83c707dd637870b0e41cdcd2f0",
        "java_tools_javac11_windows-v7.0.zip": "11d90a147919e74d11870cdd58c4ee5de3062c08d11b16aa72d3f3bbfa9497a0",
        "java_tools_javac11_darwin-v7.0.zip": "373a4226906ae9ba908550da16e133c4cd1f01b8973af82b9a2eb6903cb4d645",
        "coverage_output_generator-v2.1.zip": "96ac6bc9b9fbc67b532bcae562da1642409791e6a4b8e522f04946ee5cc3ff8e",
        "zulu11.2.3-jdk11.0.1-linux_x64.tar.gz": "232b1c3511f0d26e92582b7c3cc363be7ac633e371854ca2f2e9f2b50eb72a75",
        "zulu11.31.15-ca-jdk11.0.3-linux_aarch64.tar.gz": "3b0d91611b1bdc4d409afcf9eab4f0e7f4ae09f88fc01bd9f2b48954882ae69b",
        "zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz": "1edf366ee821e5db8e348152fcb337b28dfd6bf0f97943c270dcc6747cedb6cb",
        "zulu11.2.3-jdk11.0.1-win_x64.zip": "8e1e2b8347de6746f3fd1538840dd643201533ab113abc4ed93678e342d28aa3",
        "zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz": "f3f44b6235508e87b760bf37a49e186cc1fa4e9cd28384c4dbf5a33991921e08",
        "zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip": "059f8e3484bf07b63a8f2820d5f528f473eff1befdb1896ee4f8ff06be3b8d8f",
        "zulu11.29.3-ca-jdk11.0.2-win_x64.zip": "e1f5b4ce1b9148140fae2fcfb8a96d1c9b7eac5b8df0e13fbcad9b8561284880",
        "android_tools_pkg-0.14.tar.gz": "a3a951838448483e7af25afd10671b266cc6283104b4a2a427d31cac12cf0912",  # built at 6c63d70ef9c11a662b8323c0ae4f6d3ac53b1a60
        # bazelbuild/platforms
        "46993efdd33b73649796c5fc5c9efb193ae19d51.zip": "66184688debeeefcc2a16a2f80b03f514deac8346fe888fb7e691a52c023dd88",
        # bazelbuild/rules_java
        "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip": "bc81f1ba47ef5cc68ad32225c3d0e70b8c6f6077663835438da8d5733f917598",
        # bazelbuild/rules_cc
        "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip": "1d4dbbd1e1e9b57d40bb0ade51c9e882da7658d5bfbf22bbd15b68e7879d761f",
        # bazelbuild/rules_proto
        "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz": "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
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
        "java_tools_javac11_linux-v7.0.zip": ["https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_linux-v7.0.zip"],
        "java_tools_javac11_windows-v7.0.zip": ["https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_windows-v7.0.zip"],
        "java_tools_javac11_darwin-v7.0.zip": ["https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_darwin-v7.0.zip"],
        "coverage_output_generator-v2.1.zip": ["https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.1.zip"],
        "zulu11.2.3-jdk11.0.1-linux_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-linux_x64.tar.gz"],
        "zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz"],
        "zulu11.2.3-jdk11.0.1-win_x64.zip": ["https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-win_x64.zip"],
        "zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-linux_x64.tar.gz"],
        "zulu11.31.15-ca-jdk11.0.3-linux_aarch64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.31.15-ca-jdk11.0.3/zulu11.31.15-ca-jdk11.0.3-linux_aarch64.tar.gz"],
        "zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip": ["https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-macosx_x64.zip"],
        "zulu11.29.3-ca-jdk11.0.2-win_x64.zip": ["https://mirror.bazel.build/openjdk/azul-zulu11.29.3-ca-jdk11.0.2/zulu11.29.3-ca-jdk11.0.2-win_x64.zip"],
        "android_tools_pkg-0.14.tar.gz": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.14.tar.gz",
        ],
        # bazelbuild/platforms
        "46993efdd33b73649796c5fc5c9efb193ae19d51.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/archive/46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
            "https://github.com/bazelbuild/platforms/archive/46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
        ],
        # bazelbuild/rules_java
        "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
            "https://github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        ],
        # bazelbuild/rules_cc
        "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
            "https://github.com/bazelbuild/rules_cc/archive/8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip",
        ],
        # bazelbuild/rules_proto
        "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
        ],
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

http_archive(
    name = "java_tools_langtools_javac9",
    sha256 = "d94befcfb325a9a62aebc2052e631fde2322b4df5c82a19ed260b38ba12a0ad1",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/jdk_langtools/langtools_jdk9_v2.zip",
    ],
)

http_archive(
    name = "java_tools_langtools_javac10",
    sha256 = "0e9c9ac5ef17869de3cb8c3497c4c0d31836ef7b63efe1690506f53783adb212",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/jdk_langtools/langtools_jdk10_v2.zip",
    ],
)

http_archive(
    name = "java_tools_langtools_javac11",
    sha256 = "cf0814fa002ef3d794582bb086516d8c9ed0958f83f19799cdb08949019fe4c7",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/jdk_langtools/langtools_jdk11_v2.zip",
    ],
)

http_archive(
    name = "platforms",
    sha256 = "66184688debeeefcc2a16a2f80b03f514deac8346fe888fb7e691a52c023dd88",
    strip_prefix = "platforms-46993efdd33b73649796c5fc5c9efb193ae19d51",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/archive/46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
        "https://github.com/bazelbuild/platforms/archive/46993efdd33b73649796c5fc5c9efb193ae19d51.zip",
    ],
)

http_archive(
    name = "java_tools_langtools_javac12",
    sha256 = "99b107105165a91df82cd7cf82a8efb930d803fb7de1663cf7f780142104cd14",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/jdk_langtools/langtools_jdk12.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/android/android_remote_tools.WORKSPACE
http_archive(
    name = "android_tools_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "a3a951838448483e7af25afd10671b266cc6283104b4a2a427d31cac12cf0912",  # built at 6c63d70ef9c11a662b8323c0ae4f6d3ac53b1a60
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.14.tar.gz",
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/coverage.WORKSPACE.
http_archive(
    name = "remote_coverage_tools_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "96ac6bc9b9fbc67b532bcae562da1642409791e6a4b8e522f04946ee5cc3ff8e",
    urls = [
        "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.1.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk_linux_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk_macos_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "404e7058ff91f956612f47705efbee8e175a38b505fb1b52d8c1ea98718683de",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk_win_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "e738829017f107e7a7cd5069db979398ec3c3f03ef56122f89ba38e7374f63ed",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-win_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk_linux_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "72e7843902b0395e2d30e1e9ad2a5f05f36a4bc62529828bcbc698d54aec6022",
    strip_prefix = "jdk9-server-release-1708",
    urls = [
        # When you update this, also update the link to the source-code above.
        "https://mirror.bazel.build/openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
        "http://openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk10_linux_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "57fad3602e74c79587901d6966d3b54ef32cb811829a2552163185d5064fe9b5",
    strip_prefix = "zulu10.2+3-jdk10.0.1-linux_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-linux_x64-allmodules.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk10_macos_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "e669c9a897413d855b550b4e39d79614392e6fb96f494e8ef99a34297d9d85d3",
    strip_prefix = "zulu10.2+3-jdk10.0.1-macosx_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-macosx_x64-allmodules.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk10_win_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "c39e7700a8d41794d60985df5a20352435196e78ecbc6a2b30df7be8637bffd5",
    strip_prefix = "zulu10.2+3-jdk10.0.1-win_x64-allmodules",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-win_x64-allmodules.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk10_linux_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "b7098b7aaf6ee1ffd4a2d0371a0be26c5a5c87f6aebbe46fe9a92c90583a84be",
    strip_prefix = "jdk10-server-release-1804",
    urls = [
        # When you update this, also update the link to the source-code above.
        "https://mirror.bazel.build/openjdk.linaro.org/releases/jdk10-server-release-1804.tar.xz",
        "http://openjdk.linaro.org/releases/jdk10-server-release-1804.tar.xz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_linux_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "232b1c3511f0d26e92582b7c3cc363be7ac633e371854ca2f2e9f2b50eb72a75",
    strip_prefix = "zulu11.2.3-jdk11.0.1-linux_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_linux_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "3b0d91611b1bdc4d409afcf9eab4f0e7f4ae09f88fc01bd9f2b48954882ae69b",
    strip_prefix = "zulu11.31.15-ca-jdk11.0.3-linux_aarch64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.15-ca-jdk11.0.3/zulu11.31.15-ca-jdk11.0.3-linux_aarch64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_macos_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "1edf366ee821e5db8e348152fcb337b28dfd6bf0f97943c270dcc6747cedb6cb",
    strip_prefix = "zulu11.2.3-jdk11.0.1-macosx_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_win_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "8e1e2b8347de6746f3fd1538840dd643201533ab113abc4ed93678e342d28aa3",
    strip_prefix = "zulu11.2.3-jdk11.0.1-win_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.2.3-jdk11.0.1/zulu11.2.3-jdk11.0.1-win_x64.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remote_java_tools_linux_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "3ff465e82954a70f49982610dd63f6f651beaa83c707dd637870b0e41cdcd2f0",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_linux-v7.0.zip",
        "https://github.com/bazelbuild/java_tools/releases/download/javac11-v7.0/java_tools_javac11_linux-v7.0.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remote_java_tools_windows_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "11d90a147919e74d11870cdd58c4ee5de3062c08d11b16aa72d3f3bbfa9497a0",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_windows-v7.0.zip",
        "https://github.com/bazelbuild/java_tools/releases/download/javac11-v7.0/java_tools_javac11_windows-v7.0.zip",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remote_java_tools_darwin_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "373a4226906ae9ba908550da16e133c4cd1f01b8973af82b9a2eb6903cb4d645",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_darwin-v7.0.zip",
        "https://github.com/bazelbuild/java_tools/releases/download/javac11-v7.0/java_tools_javac11_darwin-v7.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac9_test_linux",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "e120f4a17365e7fbfc8ef0c1c24ece6668f1db295924bfe0c1b8d52caf8ad3a1",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac9/v4.0/java_tools_javac9_linux-v4.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac9_test_windows",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "5d1caf08482b72cbea9a5b1530125bc4943daa70da0345ac0a630f51c1d11c6b",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac9/v4.0/java_tools_javac9_windows-v4.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac9_test_darwin",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "03fd111111cc74b44984fb1eb945eec8ab4a91037538975bc403c26b501ee75f",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac9/v4.0/java_tools_javac9_darwin-v4.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac10_test_linux",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "d4fbed2bb22634835e42568c2d26df34fdd7281d9a7061c537f32c9970316e38",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac10/v5.0/java_tools_javac10_linux-v5.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac10_test_windows",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "ee3192ca33c1249ed785c755f84b5a989c22b4a6ca4cb1d7a37dd94104a9999d",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac10/v5.0/java_tools_javac10_windows-v5.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac10_test_darwin",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "d9311bdccb5cfcc8783b70b81a50ca72029fb35a19295dd056a7f7050d71033f",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac10/v5.0/java_tools_javac10_darwin-v5.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac11_test_linux",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "3ff465e82954a70f49982610dd63f6f651beaa83c707dd637870b0e41cdcd2f0",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_linux-v7.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac11_test_windows",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "11d90a147919e74d11870cdd58c4ee5de3062c08d11b16aa72d3f3bbfa9497a0",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_windows-v7.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac11_test_darwin",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "373a4226906ae9ba908550da16e133c4cd1f01b8973af82b9a2eb6903cb4d645",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac11/v7.0/java_tools_javac11_darwin-v7.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac12_test_linux",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "3997ee9a57b095748f1c0d084839fab2fbc72504aeb7b37b1f71c31738d330e3",
    urls = ["https://mirror.bazel.build/bazel_java_tools/releases/javac12/v3.0/java_tools_javac12_linux-v3.0.zip"],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac12_test_windows",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "cfad1718dad1fed12816748eed27ab30b9ea1268c8ce9940acf3b5b7d82d483d",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac12/v3.0/java_tools_javac12_windows-v3.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "remote_java_tools_javac12_test_darwin",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "54df966e7583bafe659e39b4103a4ce934201d969de638d071ada07d8e0c1a3a",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/releases/javac12/v3.0/java_tools_javac12_darwin-v3.0.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk9_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "45f2dfbee93b91b1468cf81d843fc6d9a47fef1f831c0b7ceff4f1eb6e6851c8",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-linux_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk9_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-macosx_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk9_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-win_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-win_x64.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk10_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "b3c2d762091a615b0c1424ebbd05d75cc114da3bf4f25a0dec5c51ea7e84146f",
    strip_prefix = "zulu10.2+3-jdk10.0.1-linux_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk10_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu10.2+3-jdk10.0.1-macosx_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk10_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu10.2+3-jdk10.0.1-win_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu10.2+3-jdk10.0.1/zulu10.2+3-jdk10.0.1-win_x64.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu11.31.11-ca-jdk11.0.3-linux_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.11-ca-jdk11.0.3/zulu11.31.11-ca-jdk11.0.3-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu11.31.11-ca-jdk11.0.3-macosx_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.11-ca-jdk11.0.3/zulu11.31.11-ca-jdk11.0.3-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu11.31.11-ca-jdk11.0.3-win_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.31.11-ca-jdk11.0.3/zulu11.31.11-ca-jdk11.0.3-win_x64.zip",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk12_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu12.2.3-ca-jdk12.0.1-linux_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu12.2.3-ca-jdk12.0.1/zulu12.2.3-ca-jdk12.0.1-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk12_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu12.2.3-ca-jdk12.0.1-macosx_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu12.2.3-ca-jdk12.0.1/zulu12.2.3-ca-jdk12.0.1-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk12_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    strip_prefix = "zulu12.2.3-ca-jdk12.0.1-win_x64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu12.2.3-ca-jdk12.0.1/zulu12.2.3-ca-jdk12.0.1-win_x64.zip",
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

register_execution_platforms("//:default_host_platform")  # buildozer: disable=positional-args

# Tools for building deb, rpm and tar files.
http_archive(
    name = "rules_pkg",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
    ],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

# Toolchains for Resource Compilation (.rc files on Windows).
load("//src/main/res:winsdk_configure.bzl", "winsdk_configure")

winsdk_configure(name = "local_config_winsdk")

load("@local_config_winsdk//:toolchains.bzl", "register_local_rc_exe_toolchains")

register_local_rc_exe_toolchains()

register_toolchains("//src/main/res:empty_rc_toolchain")
