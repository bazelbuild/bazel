load("//tools/distributions:distribution_rules.bzl", "distrib_java_import")

licenses(["reciprocal"])  #  CDDL License

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

distrib_java_import(
    name = "jaxb",
    jars = ["jaxb-api-2.3.1-patched.jar"],
    srcjar = "jaxb-api-2.3.1-sources.jar",
    enable_distributions = ["debian"],
)
