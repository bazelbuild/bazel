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
    url = "https://static.rust-lang.org/dist/rust-1.1.0-x86_64-unknown-linux-gnu.tar.gz",
    sha256 = "5a8b1c4bb254a698a69cd05734909a3933567be6996422ff53f947fd115372e6",
    build_file = "tools/build_rules/rust/rust-linux-x86_64.BUILD",
)

new_http_archive(
    name = "rust-darwin-x86_64",
    url = "https://static.rust-lang.org/dist/rust-1.1.0-x86_64-apple-darwin.tar.gz",
    sha256 = "ac802916da3f9c431377c00b864a517bc356859495b7a8a123ce2c532ee8fa83",
    build_file = "tools/build_rules/rust/rust-darwin-x86_64.BUILD",
)
