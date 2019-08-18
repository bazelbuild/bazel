load("@rules_jvm_external//:defs.bzl", "maven_install")

def rules_jmh_maven_deps(
    jmh_version = "1.21",
    repositories = ["https://repo1.maven.org/maven2"]):
    """Loads the maven dependencies of rules_jmh.
    Args:
      jmh_version: The version of the JMH library.
      repositories: A list of maven repository URLs where
        to fetch JMH from.
    """

    maven_install(
        name = "rules_jmh_maven",
        artifacts = [
            "org.openjdk.jmh:jmh-core:{}".format(jmh_version),
            "org.openjdk.jmh:jmh-generator-annprocess:{}".format(jmh_version),
        ],
        repositories = repositories,
    )

def jmh_java_benchmarks(name, srcs, deps=[], tags=[], plugins=[], main_class="org.openjdk.jmh.Main", **kwargs):
    """Builds runnable JMH benchmarks.
    This rule builds a runnable target for one or more JMH benchmarks
    specified as srcs. It takes the same arguments as java_binary,
    except for main_class.
    """
    plugin_name = "_{}_jmh_annotation_processor".format(name)
    native.java_plugin(
        name = plugin_name,
        deps = ["@rules_jmh_maven//:org_openjdk_jmh_jmh_generator_annprocess"],
        processor_class = "org.openjdk.jmh.generators.BenchmarkProcessor",
        visibility = ["//visibility:private"],
        tags = tags,
    )
    native.java_binary(
        name = name,
        srcs = srcs,
        main_class = main_class,
        deps = deps + ["@rules_jmh_maven//:org_openjdk_jmh_jmh_core"],
        plugins = plugins + [plugin_name],
        tags = tags,
        **kwargs
    )