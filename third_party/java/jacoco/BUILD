# Description:
#   JaCoCo is a free code coverage library for Java, created by the EclEmma team.

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
    srcs = glob(["*-src.jar"]) + ["LICENSE"] + [
        "//third_party:asm/asm-7.0-sources.jar",
        "//third_party:asm/asm-analysis-7.0-sources.jar",
        "//third_party:asm/asm-commons-7.0-sources.jar",
    ],
)

java_import(
    name = "agent",
    jars = ["org.jacoco.agent-%s.jar" % VERSION],
    srcjar = "org.jacoco.agent-%s-sources.jar" % VERSION,
)

java_import(
    name = "core",
    jars = ["org.jacoco.core-%s.jar" % VERSION],
    srcjar = "org.jacoco.core-%s-sources.jar" % VERSION,
    exports = [
        "//third_party/java/asm",
        "//third_party/java/asm:asm-commons",
        "//third_party/java/asm:asm-tree",
    ],
)

java_import(
    name = "report",
    jars = ["org.jacoco.report-%s.jar" % VERSION],
    srcjar = "org.jacoco.report-%s-sources.jar" % VERSION,
    exports = [
        ":core",
        "//third_party/java/asm",
    ],
)

filegroup(
    name = "blaze-agent",
    srcs = ["jacocoagent.jar"],
)
