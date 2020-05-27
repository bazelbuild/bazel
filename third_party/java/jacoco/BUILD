# Description:
#   JaCoCo is a free code coverage library for Java, created by the EclEmma team.
#
# http://search.maven.org/remotecontent?filepath=org/jacoco/jacoco/0.8.3/jacoco-0.8.3.zip

load("//tools/distributions:distribution_rules.bzl", "distrib_java_import", "distrib_jar_filegroup")

licenses(["reciprocal"])  # EPL 1.0 (Eclipse Public License)

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

VERSION = "0.8.3"

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

genrule(
    name = "jacoco_jars_zip",
    srcs = glob(["*.jar"]) + ["LICENSE"] + [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
    outs = ["jacoco_jars.zip"],
    cmd = "$(location //src:zip_files) third_party/java/jacoco $@ $(SRCS)",
    output_to_bindir = 1,
    tools = ["//src:zip_files"],
)

filegroup(
    name = "transitive_sources",
    srcs = glob(
        ["*-src.jar"],
        allow_empty = True,
    ) + ["LICENSE"] + [
        "//third_party:asm/asm-8.0-sources.jar",
        "//third_party:asm/asm-analysis-8.0-sources.jar",
        "//third_party:asm/asm-commons-8.0-sources.jar",
    ],
)

distrib_java_import(
    name = "agent",
    jars = ["org.jacoco.agent-%s.jar" % VERSION],
    srcjar = "org.jacoco.agent-%s-sources.jar" % VERSION,
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "agent-%s" % VERSION,
    jars = ["org.jacoco.agent-%s.jar" % VERSION],
    srcjar = "org.jacoco.agent-%s-sources.jar" % VERSION,
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "core",
    jars = ["org.jacoco.core-%s.jar" % VERSION],
    srcjar = "org.jacoco.core-%s-sources.jar" % VERSION,
    enable_distributions = ["debian"],
    exports = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
)

distrib_java_import(
    name = "core-%s" % VERSION,
    jars = ["org.jacoco.core-%s.jar" % VERSION],
    srcjar = "org.jacoco.core-%s-sources.jar" % VERSION,
    enable_distributions = ["debian"],
    exports = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
)

distrib_jar_filegroup(
    name = "core-jars",
    srcs = ["org.jacoco.core-%s.jar" % VERSION],
    enable_distributions = ["debian"],
)

distrib_jar_filegroup(
    name = "core-jars-%s" % VERSION,
    srcs = ["org.jacoco.core-%s.jar" % VERSION],
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "report",
    jars = ["org.jacoco.report-%s.jar" % VERSION],
    srcjar = "org.jacoco.report-%s-sources.jar" % VERSION,
    enable_distributions = ["debian"],
    exports = [
        ":core",
        "//third_party:asm",
    ],
)

distrib_java_import(
    name = "report-%s" % VERSION,
    jars = ["org.jacoco.report-%s.jar" % VERSION],
    srcjar = "org.jacoco.report-%s-sources.jar" % VERSION,
    enable_distributions = ["debian"],
    exports = [
        ":core-%s" % VERSION,
        "//third_party:asm",
    ],
)

distrib_java_import(
    name = "blaze-agent",
    jars = ["jacocoagent-%s.jar" % VERSION],
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "blaze-agent-%s" % VERSION,
    jars = ["jacocoagent-%s.jar" % VERSION],
    enable_distributions = ["debian"],
)
