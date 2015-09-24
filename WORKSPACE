# For src/tools/dash support.

new_http_archive(
    name = "appengine-java",
    url = "http://central.maven.org/maven2/com/google/appengine/appengine-java-sdk/1.9.23/appengine-java-sdk-1.9.23.zip",
    sha256 = "05e667036e9ef4f999b829fc08f8e5395b33a5a3c30afa9919213088db2b2e89",
    build_file = "tools/build_rules/appengine/appengine.BUILD",
)

bind(
    name = "appengine/java/sdk",
    actual = "@appengine-java//:sdk",
)

bind(
    name = "appengine/java/api",
    actual = "@appengine-java//:api",
)

bind(
    name = "appengine/java/jars",
    actual = "@appengine-java//:jars",
)

maven_jar(
    name = "javax-servlet-api",
    artifact = "javax.servlet:servlet-api:2.5",
)

maven_jar(
    name = "commons-lang",
    artifact = "commons-lang:commons-lang:2.6",
)

bind(
    name = "javax/servlet/api",
    actual = "//tools/build_rules/appengine:javax.servlet.api",
)

maven_jar(
    name = "easymock",
    artifact = "org.easymock:easymock:3.1",
)

new_http_archive(
    name = "rust-linux-x86_64",
    url = "https://static.rust-lang.org/dist/rust-1.3.0-x86_64-unknown-linux-gnu.tar.gz",
    sha256 = "fa755b6331ff7554e6e8545ee20af7897b0adc65f471dd24ae8a467a944755b4",
    build_file = "tools/build_rules/rust/rust.BUILD",
)

new_http_archive(
    name = "rust-darwin-x86_64",
    url = "https://static.rust-lang.org/dist/rust-1.3.0-x86_64-apple-darwin.tar.gz",
    sha256 = "bfeac876e22cc5fe63a250644ce1a6f3892c13a5461131a881419bd06fcb2011",
    build_file = "tools/build_rules/rust/rust.BUILD",
)

# In order to run the Android integration tests, uncomment these rules, point
# them to the Android NDK and the SDK, and point the bind rules under them
# to @repository//:files .
# android_sdk_repository(
#     name = "androidsdk",
#     path = "/path/to/sdk",
#     build_tools_version = "21.1.1",
#     api_level = 19,
# )

# android_ndk_repository(
#     name = "androidndk",
#     path = "/path/to/ndk",
#     api_level = 19,
# )

bind(
    name = "android_sdk_for_testing",
    actual = "//:dummy",
)

bind(
    name = "android_ndk_for_testing",
    actual = "//:dummy",
)

new_http_archive(
    name = "dmd-linux-x86_64",
    url = "http://downloads.dlang.org/releases/2.x/2.067.1/dmd.2.067.1.linux.zip",
    sha256 = "a5014886773853b4a42df19ee9591774cf281d33fbc511b265df30ba832926cd",
    build_file = "tools/build_defs/d/dmd.BUILD",
)

new_http_archive(
    name = "dmd-darwin-x86_64",
    url = "http://downloads.dlang.org/releases/2.x/2.067.1/dmd.2.067.1.osx.zip",
    sha256 = "aa76bb83c38b3f7495516eb08977fc9700c664d7a945ba3ac3c0004a6a8509f2",
    build_file = "tools/build_defs/d/dmd.BUILD",
)

new_git_repository(
    name = "jsonnet",
    remote = "https://github.com/google/jsonnet.git",
    tag = "v0.8.0",
    build_file = "tools/build_defs/jsonnet/jsonnet.BUILD",
)

new_http_archive(
    name = "libsass",
    url = "https://github.com/sass/libsass/archive/3.3.0-beta1.tar.gz",
    sha256 = "6a4da39cc0b585f7a6ee660dc522916f0f417c890c3c0ac7ebbf6a85a16d220f",
    build_file = "tools/build_defs/sass/libsass.BUILD",
)

new_http_archive(
    name = "sassc",
    url = "https://github.com/sass/sassc/archive/3.3.0-beta1.tar.gz",
    sha256 = "87494218eea2441a7a24b40f227330877dbba75c5fa9014ac6188711baed53f6",
    build_file = "tools/build_defs/sass/sassc.BUILD",
)
