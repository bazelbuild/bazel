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
    url = "https://static.rust-lang.org/dist/rust-1.2.0-x86_64-unknown-linux-gnu.tar.gz",
    sha256 = "2311420052e06b3e698ce892924ec40890a8ff0499902e7fc5350733187a1531",
    build_file = "tools/build_rules/rust/rust-linux-x86_64.BUILD",
)

new_http_archive(
    name = "rust-darwin-x86_64",
    url = "https://static.rust-lang.org/dist/rust-1.2.0-x86_64-apple-darwin.tar.gz",
    sha256 = "0d471e672fac5a450ae5507b335fda2efc0b22ea9fb7f215c6a9c466dafa2661",
    build_file = "tools/build_rules/rust/rust-darwin-x86_64.BUILD",
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
