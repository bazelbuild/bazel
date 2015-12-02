new_http_archive(
    name = "rust-linux-x86_64",
    url = "https://static.rust-lang.org/dist/rust-1.4.0-x86_64-unknown-linux-gnu.tar.gz",
    strip_prefix = "rust-1.4.0-x86_64-unknown-linux-gnu",
    sha256 = "2de2424b50ca2ab3a67c495b6af03c720801a2928ad30884438ad0f5436ac51d",
    build_file = "tools/build_rules/rust/rust.BUILD",
)

new_http_archive(
    name = "rust-darwin-x86_64",
    url = "https://static.rust-lang.org/dist/rust-1.4.0-x86_64-apple-darwin.tar.gz",
    strip_prefix = "rust-1.4.0-x86_64-apple-darwin",
    sha256 = "7256617aec7c106be2aa3c5df0a2e613b13ec55e6237ab612bb4164719e09e21",
    build_file = "tools/build_rules/rust/rust.BUILD",
)

# In order to run the Android integration tests, uncomment these rules, point
# them to the Android NDK and the SDK, and point the bind rules under them
# to @repository//:files .
# android_sdk_repository(
#     name = "androidsdk",
#     path = "/path/to/sdk",
#     # Available versions are under /path/to/sdk/build-tools/.
#     build_tools_version = "21.1.1",
#     # Available versions are under /path/to/sdk/platforms/.
#     api_level = 19,
# )

# android_ndk_repository(
#     name = "androidndk",
#     path = "/path/to/ndk",
#     api_level = 19,  # Set this to the SDK's api_level.
# )

bind(
    name = "android_sdk_for_testing",
    # Uncomment and delete the //:dummy line to run integration tests.
    #   actual = "@androidsdk//:files",
    actual = "//:dummy",
)

bind(
    name = "android_ndk_for_testing",
    # Uncomment and delete the //:dummy line to run integration tests.
    #   actual = "@androidndk//:files",
    actual = "//:dummy",
)

new_http_archive(
    name = "dmd-linux-x86_64",
    build_file = "tools/build_defs/d/dmd.BUILD",
    sha256 = "a5014886773853b4a42df19ee9591774cf281d33fbc511b265df30ba832926cd",
    url = "http://downloads.dlang.org/releases/2.x/2.067.1/dmd.2.067.1.linux.zip",
)

new_http_archive(
    name = "dmd-darwin-x86_64",
    build_file = "tools/build_defs/d/dmd.BUILD",
    sha256 = "aa76bb83c38b3f7495516eb08977fc9700c664d7a945ba3ac3c0004a6a8509f2",
    url = "http://downloads.dlang.org/releases/2.x/2.067.1/dmd.2.067.1.osx.zip",
)

git_repository(
    name = "jsonnet",
    remote = "https://github.com/google/jsonnet.git",
    tag = "v0.8.1",
)

new_http_archive(
    name = "libsass",
    build_file = "tools/build_defs/sass/libsass.BUILD",
    sha256 = "6a4da39cc0b585f7a6ee660dc522916f0f417c890c3c0ac7ebbf6a85a16d220f",
    url = "https://github.com/sass/libsass/archive/3.3.0-beta1.tar.gz",
)

new_http_archive(
    name = "sassc",
    build_file = "tools/build_defs/sass/sassc.BUILD",
    sha256 = "87494218eea2441a7a24b40f227330877dbba75c5fa9014ac6188711baed53f6",
    url = "https://github.com/sass/sassc/archive/3.3.0-beta1.tar.gz",
)

bind(
    name = "go_prefix",
    actual = "//:go_prefix",
)

new_http_archive(
    name = "golang-darwin-amd64",
    build_file = "tools/build_rules/go/toolchain/BUILD.go-toolchain",
    sha256 = "e94487b8cd2e0239f27dc51e6c6464383b10acb491f753584605e9b28abf48fb",
    url = "https://storage.googleapis.com/golang/go1.5.1.darwin-amd64.tar.gz",
)

new_http_archive(
    name = "golang-linux-amd64",
    build_file = "tools/build_rules/go/toolchain/BUILD.go-toolchain",
    sha256 = "2593132ca490b9ee17509d65ee2cd078441ff544899f6afb97a03d08c25524e7",
    url = "https://storage.googleapis.com/golang/go1.5.1.linux-amd64.tar.gz",
)

new_http_archive(
    name = "nunit",
    build_file = "tools/build_defs/dotnet/nunit.BUILD",
    sha256 = "1bd925514f31e7729ccde40a38a512c2accd86895f93465f3dfe6d0b593d7170",
    type = "zip",
    url = "https://github.com/nunit/nunitv2/releases/download/2.6.4/NUnit-2.6.4.zip",
)
