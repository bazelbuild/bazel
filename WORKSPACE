load("/tools/build_defs/jsonnet/jsonnet", "jsonnet_repositories")
load("/tools/build_rules/rust/rust", "rust_repositories")

jsonnet_repositories()
rust_repositories()

# Protobuf expects an //external:python_headers label which would contain the
# Python headers if fast Python protos is enabled. Since we are not using fast
# Python protos, bind python_headers to a dummy target.
bind(
    name = "python_headers",
    actual = "//:dummy",
)

# Bind to dummy targets if no android SDK/NDK is present.
bind(
    name = "android_sdk_for_testing",
    actual = "//:dummy",
)

bind(
    name = "android_ndk_for_testing",
    actual = "//:dummy",
)

# For building for JDK 7
# This is a JDK 7 JavaBuilder from release 0.1.0.
# Do not use the skylark bootstrapped version of JavaBuilder
http_file(
    name = "io_bazel_javabuilder_jdk7",
    # This was built from Bazel 0.1.0 (after ./compile.sh) and JDK 7.
    url = "https://storage.googleapis.com/bazel/0.1.0/JavaBuilder_deploy.jar",
    sha256 = "374f07be2f73ec170ef6bdd8057530e6842cb0cbc7a240caccbd7b80be239366",
)

http_file(
    name = "io_bazel_javac_jdk7",
    url = "https://github.com/bazelbuild/bazel/blob/0.1.0/third_party/java/jdk/langtools/javac.jar?raw=true",
    sha256 = "e063cf60cf9a1d24acd043f55dc605da559b913793c6a4e51c5ddbc4f5d494df",
)

# For tools/cpp/test/...
load("//tools/cpp/test:docker_repository.bzl", "docker_repository")
docker_repository()

# In order to run the Android integration tests, run
# scripts/workspace_user.sh and uncomment the next two lines.
# load("/WORKSPACE.user", "android_repositories")
# android_repositories()
