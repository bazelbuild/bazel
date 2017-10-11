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
    sha256 = "529ee99e6f0e3f88edef61aeae4f13dc6e5eb8183993ced191338422b0e1fbeb",
    strip_prefix = "j2objc-2.0.3",
)

# For src/test/docker/...
load("//src/test/docker:docker_repository.bzl", "docker_repository")
docker_repository()
load("//src/test/docker:flavours.bzl", "pull_images_for_docker_tests")
pull_images_for_docker_tests()

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

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
http_file(
    name = "openjdk_linux",
    sha256 = "7e6284739c0e5b7142bc7a9adc61ced70dc5bb26b130b582b18e809013bcb251",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.23.0.3-jdk8.0.144/zulu8.23.0.3-jdk8.0.144-linux_x64.tar.gz",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.23.0.3-jdk8.0.144/zulu8.23.0.3-jdk8.0.144-linux_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu8.23.0.3-jdk8.0.144-linux_x64.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos",
    sha256 = "ff533364c9cbd3b271ab5328efe28e2dd6d7bae5b630098a5683f742ecf0709d",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.23.0.3-jdk8.0.144/zulu8.23.0.3-jdk8.0.144-macosx_x64.zip",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.23.0.3-jdk8.0.144/zulu8.23.0.3-jdk8.0.144-macosx_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu8.23.0.3-jdk8.0.144-macosx_x64.zip",
    ],
)

http_file(
    name = "openjdk_win",
    sha256 = "f1d9d3341ef7c8c9baff3597953e99a6a7c64f8608ee62c03fdd7574b7655c02",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.23.0.3-jdk8.0.144/zulu8.23.0.3-jdk8.0.144-win_x64.zip",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.23.0.3-jdk8.0.144/zulu8.23.0.3-jdk8.0.144-win_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu8.23.0.3-jdk8.0.144-win_x64.zip",
    ],
)

http_archive(
    name = "bazel_toolchains",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/b2b4b38433bf2d1159360855ea4004378308711b.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/b2b4b38433bf2d1159360855ea4004378308711b.tar.gz",
    ],
    strip_prefix = "bazel-toolchains-b2b4b38433bf2d1159360855ea4004378308711b",
    sha256 = "46187270ca04ff8109980f45c3438fabfe48695e163789096eb82ee097ffe685",
)

http_archive(
    name = "com_googlesource_code_re2",
    urls = [
        "https://github.com/google/re2/archive/2017-08-01.tar.gz",
    ],
    strip_prefix = "re2-2017-08-01",
    sha256 = "938723dc197125392698c5fcf41acb74877866ff140b81fd50b7314bf26f1636",
)
