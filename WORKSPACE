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

# For src/test/docker/...
load("//src/test/docker:docker_repository.bzl", "docker_repository")

docker_repository()

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
    build_file = "./third_party/protobuf/3.2.0/BUILD",
    path = "./third_party/protobuf/3.2.0/",
)

new_local_repository(
    name = "com_google_protobuf_java",
    build_file = "./third_party/protobuf/3.2.0/com_google_protobuf_java.BUILD",
    path = "./third_party/protobuf/3.2.0/",
)

new_local_repository(
    name = "googleapis",
    path = "./third_party/googleapis/",
    build_file = "./third_party/googleapis/BUILD",
)

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
http_file(
    name = "openjdk_linux",
    sha256 = "17218c6bdd608b5714ffba9d5e28522bb2efc309266ba46232b8b918e6e62133",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.21.0.1-jdk8.0.131/zulu8.21.0.1-jdk8.0.131-linux_x64.tar.gz",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.21.0.1-jdk8.0.131/zulu8.21.0.1-jdk8.0.131-linux_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu8.21.0.1-jdk8.0.131-linux_x64.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos",
    sha256 = "87575cbe5dc98ae4326c3c786dbe53515030f832451af910c11ce1b67e5b0977",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.21.0.1-jdk8.0.131/zulu8.21.0.1-jdk8.0.131-macosx_x64.zip",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.21.0.1-jdk8.0.131/zulu8.21.0.1-jdk8.0.131-macosx_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu8.21.0.1-jdk8.0.131-macosx_x64.zip",
    ],
)

http_file(
    name = "openjdk_win",
    sha256 = "474120caa9748e3512e2cd38d304a63d70cbf747ca2f8aa915f0a880ed807eb4",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu-8.21.0.1-jdk8.0.131/zulu8.21.0.1-jdk8.0.131-win_x64.zip",
        "https://bazel-mirror.storage.googleapis.com/openjdk/azul-zulu-8.21.0.1-jdk8.0.131/zulu8.21.0.1-jdk8.0.131-win_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu8.21.0.1-jdk8.0.131-win_x64.zip",
    ],
)
