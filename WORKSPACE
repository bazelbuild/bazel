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
    url = "https://github.com/google/desugar_jdk_libs/archive/f5e6d80c6b4ec6b0a46603f72b015d45cf3c11cd.zip",
    # Computed using "shasum -a 256 <zip>"
    sha256 = "c80f3f3d442d8a6ca7adc83f90ecd638c3864087fdd6787ffac070b6f1cc8f9b",
    strip_prefix = "desugar_jdk_libs-f5e6d80c6b4ec6b0a46603f72b015d45cf3c11cd",
)

load("//:distdir.bzl", "distdir_tar")
distdir_tar(
    name = "additional_distfiles",
    dirname = "derived/distdir",
    archives = ["f5e6d80c6b4ec6b0a46603f72b015d45cf3c11cd.zip"],
    urls = {
        "f5e6d80c6b4ec6b0a46603f72b015d45cf3c11cd.zip" :
        ["https://github.com/google/desugar_jdk_libs/archive/f5e6d80c6b4ec6b0a46603f72b015d45cf3c11cd.zip"],
    },
    sha256 = {
        "f5e6d80c6b4ec6b0a46603f72b015d45cf3c11cd.zip" :
        "c80f3f3d442d8a6ca7adc83f90ecd638c3864087fdd6787ffac070b6f1cc8f9b",
  },
)

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
http_file(
    name = "openjdk_linux",
    sha256 = "85b81652b3fe8cfb0a2cfb835988672bde7844f613dae5b9487b5b44921a1afd",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.28.0.1-jdk8.0.163/zulu8.28.0.1-jdk8.0.163-linux_x64.tar.gz",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.28.0.1-jdk8.0.163/zulu8.28.0.1-jdk8.0.163-linux_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu8.28.0.1-jdk8.0.163-linux_x64.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos",
    sha256 = "292a44695e708a8822b3d0a462faf32887ddb80b56aa91e7598fd4b6f6341f69",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.28.0.1-jdk8.0.163/zulu8.28.0.1-jdk8.0.163-macosx_x64.tar.gz",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.28.0.1-jdk8.0.163/zulu8.28.0.1-jdk8.0.163-macosx_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu8.28.0.1-jdk8.0.163-macosx_x64.tar.gz",
    ],
)

http_file(
    name = "openjdk_win",
    sha256 = "0c5ea3634ae7d7851630cf61fdd6344cce110cf9246593345e49fc430bb39442",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.28.0.1-jdk8.0.163/zulu8.28.0.1-jdk8.0.163-win_x64.zip",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.28.0.1-jdk8.0.163/zulu8.28.0.1-jdk8.0.163-win_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu8.28.0.1-jdk8.0.163-win_x64.zip",
    ],
)

http_archive(
    name = "bazel_toolchains",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/44200e0c026d86c53470d107b3697a3e46469c43.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/44200e0c026d86c53470d107b3697a3e46469c43.tar.gz",
    ],
    strip_prefix = "bazel-toolchains-44200e0c026d86c53470d107b3697a3e46469c43",
    sha256 = "699b55a6916c687f4b7dc092dbbf5f64672cde0dc965f79717735ec4e5416556",
)

http_archive(
    name = "com_googlesource_code_re2",
    urls = [
        "https://github.com/google/re2/archive/2017-12-01.tar.gz",
    ],
    strip_prefix = "re2-2017-12-01",
    sha256 = "62797e7cd7cc959419710cd25b075b5f5b247da0e8214d47bf5af9b32128fb0d",
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
)
