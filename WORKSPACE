workspace(name = "io_bazel")

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
    url = "https://github.com/google/j2objc/releases/download/2.0.3/j2objc-2.0.3.zip",
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

new_local_repository(
    name = "googleapis",
    build_file = "./third_party/googleapis/BUILD",
    path = "./third_party/googleapis/",
)

new_local_repository(
    name = "remoteapis",
    build_file = "./third_party/remoteapis/BUILD.bazel",
    path = "./third_party/remoteapis/",
)

http_archive(
    name = "desugar_jdk_libs",
    # Computed using "shasum -a 256 <zip>"
    sha256 = "43b8fcc56a180e178d498f375fbeb95e8b65b9bf6c2da91ae3ae0332521a1a12",
    strip_prefix = "desugar_jdk_libs-fd937f4180c1b557805219af4482f1a27eb0ff2b",
    url = "https://github.com/google/desugar_jdk_libs/archive/fd937f4180c1b557805219af4482f1a27eb0ff2b.zip",
)

load("//:distdir.bzl", "distdir_tar")

distdir_tar(
    name = "additional_distfiles",
    archives = [
        "fd937f4180c1b557805219af4482f1a27eb0ff2b.zip",
        "7490380c6bbf9a5a060df78dc2222e7de6ffae5c.tar.gz",
    ],
    dirname = "derived/distdir",
    sha256 = {
        "fd937f4180c1b557805219af4482f1a27eb0ff2b.zip": "43b8fcc56a180e178d498f375fbeb95e8b65b9bf6c2da91ae3ae0332521a1a12",
        "7490380c6bbf9a5a060df78dc2222e7de6ffae5c.tar.gz": "3528fc6012a78da6291c00854373ea43f7f8b6c4046320be5f0884f5b3385b14",
    },
    urls = {
        "fd937f4180c1b557805219af4482f1a27eb0ff2b.zip": ["https://github.com/google/desugar_jdk_libs/archive/fd937f4180c1b557805219af4482f1a27eb0ff2b.zip"],
        "7490380c6bbf9a5a060df78dc2222e7de6ffae5c.tar.gz": ["https://github.com/bazelbuild/bazel-skylib/archive/7490380c6bbf9a5a060df78dc2222e7de6ffae5c.tar.gz"],
    },
)

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
http_file(
    name = "openjdk_linux",
    sha256 = "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
    ],
)

# Used by CI to test Bazel on platforms without an installed system JDK.
# TODO(twerth): Migrate to @remotejdk when https://github.com/bazelbuild/bazel/pull/6216 is merged.
new_http_archive(
    name = "openjdk_linux_archive",
    sha256 = "f27cb933de4f9e7fe9a703486cf44c84bc8e9f138be0c270c9e5716a32367e87",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules.tar.gz",
    ],
    strip_prefix = "zulu9.0.7.1-jdk9.0.7-linux_x64-allmodules",
    build_file_content = "java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])",
)

http_file(
    name = "openjdk_macos",
    sha256 = "404e7058ff91f956612f47705efbee8e175a38b505fb1b52d8c1ea98718683de",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-macosx_x64-allmodules.tar.gz",
    ],
)

http_file(
    name = "openjdk_win",
    sha256 = "e738829017f107e7a7cd5069db979398ec3c3f03ef56122f89ba38e7374f63ed",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-9.0.7.1-jdk9.0.7/zulu9.0.7.1-jdk9.0.7-win_x64-allmodules.zip",
    ],
)

# The source-code for this OpenJDK can be found at:
# https://openjdk.linaro.org/releases/jdk9-src-1708.tar.xz
http_file(
    name = "openjdk_linux_aarch64",
    sha256 = "72e7843902b0395e2d30e1e9ad2a5f05f36a4bc62529828bcbc698d54aec6022",
    urls = [
        # When you update this, also update the link to the source-code above.
        "http://openjdk.linaro.org/releases/jdk9-server-release-1708.tar.xz",
    ],
)

http_archive(
    name = "bazel_toolchains",
    sha256 = "fa1459abc7d89db728da424176f5f424e78cb8ad7a3d03d8bfa0c5c4a56b7398",
    strip_prefix = "bazel-toolchains-42619b5476b7c8a2f5117f127d5772cc46da2d1d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/42619b5476b7c8a2f5117f127d5772cc46da2d1d.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/42619b5476b7c8a2f5117f127d5772cc46da2d1d.tar.gz",
    ],
)

# We're pinning to a commit because this project does not have a recent release.
# Nothing special about this commit, though.
http_archive(
    name = "com_google_googletest",
    sha256 = "313a16fba8f0be8ee20ba9883e044556044cbd1ae6cea532473d163a843ef991",
    strip_prefix = "googletest-dfa853b63d17c787914b663b50c2095a0c5b706e",
    urls = [
        "https://github.com/google/googletest/archive/dfa853b63d17c787914b663b50c2095a0c5b706e.tar.gz",
    ],
)

# For src/test/shell/bazel:bazel_sandboxing_test
http_file(
    name = "mount_path_toolchain",
    sha256 = "dd8088d3543a86fd91a9ccde6e40102aff6eaf3d048aa73cc18eff05cc2053d5",
    url = "https://asci-toolchain.appspot.com.storage.googleapis.com/toolchain-testing/mount_path_toolchain.tar.gz",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "3528fc6012a78da6291c00854373ea43f7f8b6c4046320be5f0884f5b3385b14",
    strip_prefix = "bazel-skylib-7490380c6bbf9a5a060df78dc2222e7de6ffae5c",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/archive/7490380c6bbf9a5a060df78dc2222e7de6ffae5c.tar.gz",
    ],
)

http_archive(
    name = "skydoc",
    sha256 = "cfbfcc107f5c9853dc5b2b81f1fe90fc326bd1c61f76c9aac2b4201dff75b91d",
    strip_prefix = "skydoc-d34c44c3f4102eb94beaf2636c6cf532f0ec1ee8",
    urls = [
        "https://github.com/bazelbuild/skydoc/archive/d34c44c3f4102eb94beaf2636c6cf532f0ec1ee8.tar.gz",
    ],
)
