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

# Android databinding processing requires api, core, and impl, however
# jaxb-core-2.3.1.jar is missing on maven...
# https://mvnrepository.com/artifact/com.sun.xml.bind/jaxb-core/2.3.1
# does not actually have any jars. So, use 2.3.0 instead, which works.
java_library(
    name = "jaxb-2.3.0",
    exports = [
      ":jaxb-api-2.3.0",
      ":jaxb-core-2.3.0",
      ":jaxb-impl-2.3.0",
      "//third_party:javax_activation",
    ],
    visibility = ["//third_party/java/android_databinding/v3_4_0:__pkg__"],
)

java_import(
    name = "jaxb-api-2.3.0",
    jars = ["2.3.0/jaxb-api-2.3.0.jar"],
    srcjar = "2.3.0/jaxb-api-2.3.0-sources.jar",
    visibility = ["//visibility:private"],
)

java_import(
    name = "jaxb-core-2.3.0",
    jars = ["2.3.0/jaxb-core-2.3.0.jar"],
    srcjar = "2.3.0/jaxb-core-2.3.0-sources.jar",
    visibility = ["//visibility:private"],
)

java_import(
    name = "jaxb-impl-2.3.0",
    jars = ["2.3.0/jaxb-impl-2.3.0.jar"],
    srcjar = "2.3.0/jaxb-impl-2.3.0-sources.jar",
    visibility = ["//visibility:private"],
)
