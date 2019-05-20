# Description:
#   JaCoCo is a free code coverage library for Java, created by the EclEmma team.
#
# http://search.maven.org/remotecontent?filepath=org/jacoco/jacoco/0.8.3/jacoco-0.8.3.zip

licenses(["reciprocal"])  # EPL 1.0 (Eclipse Public License)

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

NEW_VERSION = "0.8.3"
OLD_VERSION = "0.7.5.201505241946"

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
    jars = ["org.jacoco.agent-%s.jar" % OLD_VERSION],
    srcjar = "org.jacoco.agent-%s-sources.jar" % OLD_VERSION,
)

java_import(
    name = "agent-%s" % NEW_VERSION,
    jars = ["org.jacoco.agent-%s.jar" % NEW_VERSION],
    srcjar = "org.jacoco.agent-%s-sources.jar" % NEW_VERSION,
)

java_import(
    name = "core",
    jars = ["org.jacoco.core-%s.jar" % OLD_VERSION],
    srcjar = "org.jacoco.core-%s-sources.jar" % OLD_VERSION,
    exports = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
)

java_import(
    name = "core-%s" % NEW_VERSION,
    jars = ["org.jacoco.core-%s.jar" % NEW_VERSION],
    srcjar = "org.jacoco.core-%s-sources.jar" % NEW_VERSION,
    exports = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
)

filegroup(
    name = "core-jars",
    srcs = ["org.jacoco.core-%s.jar" % OLD_VERSION],
)

filegroup(
    name = "core-jars-%s" % NEW_VERSION,
    srcs = ["org.jacoco.core-%s.jar" % NEW_VERSION],
)

java_import(
    name = "report",
    jars = ["org.jacoco.report-%s.jar" % OLD_VERSION],
    srcjar = "org.jacoco.report-%s-sources.jar" % OLD_VERSION,
    exports = [
        ":core",
        "//third_party:asm",
    ],
)

java_import(
    name = "report-%s" % NEW_VERSION,
    jars = ["org.jacoco.report-%s.jar" % NEW_VERSION],
    srcjar = "org.jacoco.report-%s-sources.jar" % NEW_VERSION,
    exports = [
        ":core-%s" % NEW_VERSION,
        "//third_party:asm",
    ],
)

java_import(
    name = "blaze-agent",
    jars = ["jacocoagent-%s.jar" % OLD_VERSION],
)

java_import(
    name = "blaze-agent-%s" % NEW_VERSION,
    jars = ["jacocoagent-%s.jar" % NEW_VERSION],
)

java_import(
    name = "blaze-agent-neverlink",
    jars = ["jacocoagent-%s.jar"  % OLD_VERSION],
    neverlink = 1,
)

java_import(
    name = "blaze-agent-neverlink-%s" % NEW_VERSION,
    jars = ["jacocoagent-%s.jar"  % NEW_VERSION],
    neverlink = 1,
)
