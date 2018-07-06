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
    url = "https://github.com/google/j2objc/releases/download/2.0.3/j2objc-2.0.3.zip",
    # Computed using "shasum -a 256 j2objc-2.0.3.zip"
    sha256 = "a36bac432d0dbd8c98249e484b2b69dd5720afa4abb58711a3c3def1c0bfa21d",
    strip_prefix = "j2objc-2.0.3",
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
    build_file = "./third_party/protobuf/3.4.0/BUILD",
    path = "./third_party/protobuf/3.4.0/",
)

new_local_repository(
    name = "com_google_protobuf_cc",
    build_file = "./third_party/protobuf/3.4.0/BUILD",
    path = "./third_party/protobuf/3.4.0/",
)

new_local_repository(
    name = "com_google_protobuf_java",
    build_file = "./third_party/protobuf/3.4.0/com_google_protobuf_java.BUILD",
    path = "./third_party/protobuf/3.4.0/",
)

new_local_repository(
    name = "googleapis",
    path = "./third_party/googleapis/",
    build_file = "./third_party/googleapis/BUILD",
)

http_archive(
    name = "desugar_jdk_libs",
    url = "https://github.com/google/desugar_jdk_libs/archive/fd937f4180c1b557805219af4482f1a27eb0ff2b.zip",
    # Computed using "shasum -a 256 <zip>"
    sha256 = "43b8fcc56a180e178d498f375fbeb95e8b65b9bf6c2da91ae3ae0332521a1a12",
    strip_prefix = "desugar_jdk_libs-fd937f4180c1b557805219af4482f1a27eb0ff2b",
)

load("//:distdir.bzl", "distdir_tar")
distdir_tar(
    name = "additional_distfiles",
    dirname = "derived/distdir",
    archives = ["fd937f4180c1b557805219af4482f1a27eb0ff2b.zip"],
    urls = {
        "fd937f4180c1b557805219af4482f1a27eb0ff2b.zip" :
        ["https://github.com/google/desugar_jdk_libs/archive/fd937f4180c1b557805219af4482f1a27eb0ff2b.zip"],
    },
    sha256 = {
        "fd937f4180c1b557805219af4482f1a27eb0ff2b.zip" :
        "43b8fcc56a180e178d498f375fbeb95e8b65b9bf6c2da91ae3ae0332521a1a12",
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

http_file(
    name = "openjdk_macos",
    sha256 = "7a1e769a433948b3d1215911730c17fc2f2710348df18aa966b38c39e4cfe444",
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

http_archive(
  name = "bazel_toolchains",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/2cec6c9f6d12224e93d9b3f337b24e41602de3ba.tar.gz",
    "https://github.com/bazelbuild/bazel-toolchains/archive/2cec6c9f6d12224e93d9b3f337b24e41602de3ba.tar.gz",
  ],
  strip_prefix = "bazel-toolchains-2cec6c9f6d12224e93d9b3f337b24e41602de3ba",
  sha256 = "9b8d85b61d8945422e86ac31e4d4d2d967542c080d1da1b45364da7fd6bdd638",
)

# We're pinning to a commit because this project does not have a recent release.
# Nothing special about this commit, though.
http_archive(
    name = "com_google_googletest",
    urls = [
        "https://github.com/google/googletest/archive/dfa853b63d17c787914b663b50c2095a0c5b706e.tar.gz",
    ],
    strip_prefix = "googletest-dfa853b63d17c787914b663b50c2095a0c5b706e",
    sha256 = "313a16fba8f0be8ee20ba9883e044556044cbd1ae6cea532473d163a843ef991",
)

# For src/test/shell/bazel:bazel_sandboxing_test
http_file(
  name = 'mount_path_toolchain',
  url = 'https://asci-toolchain.appspot.com.storage.googleapis.com/toolchain-testing/mount_path_toolchain.tar.gz',
  sha256 = "dd8088d3543a86fd91a9ccde6e40102aff6eaf3d048aa73cc18eff05cc2053d5",
)
