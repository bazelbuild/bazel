# External dependencies for the java_* rules.

new_local_repository(
    name = "local-jdk",
    path = DEFAULT_SERVER_JAVABASE,
    build_file = __embedded_dir__ + "/jdk.BUILD",
)

bind(
    name = "bootclasspath",
    actual = "@local-jdk//:bootclasspath",
)
bind(
    name = "extdir",
    actual = "@local-jdk//:extdir",
)
bind(
    name = "langtools",
    actual = "@local-jdk//:langtools",
)
bind(
    name = "langtools-neverlink",
    actual = "@local-jdk//:langtools-neverlink",
)
bind(
    name = "jni_header",
    actual = "@local-jdk//:jni_header",
)

bind(
    name = "jni_md_header-darwin",
    actual = "@local-jdk//:jni_md_header-darwin",
)

bind(
    name = "jni_md_header-linux",
    actual = "@local-jdk//:jni_md_header-linux",
)

bind(
    name = "java",
    actual = "@local-jdk//:java",
)

bind(
    name = "jdk-default",
    actual = "@local-jdk//:jdk-default",
)
