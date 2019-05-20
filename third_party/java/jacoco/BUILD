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

[
    java_import(
        name = "agent-%s" % version,
        jars = ["org.jacoco.agent-%s.jar" % version],
        srcjar = "org.jacoco.agent-%s-sources.jar" % version,
    )
    for version in [NEW_VERSION, OLD_VERSION]
]

[
    java_import(
        name = "core-%s" % s,
        jars = ["org.jacoco.core-%s.jar" % version],
        srcjar = "org.jacoco.core-%s-sources.jar" % version,
        exports = [
            "//third_party:asm",
            "//third_party:asm-commons",
            "//third_party:asm-tree",
        ],
    )
    for version in [NEW_VERSION, OLD_VERSION]
]

[
    filegroup(
        name = "core-jars-%s" % version,
        srcs = ["org.jacoco.core-%s.jar" % version],
    )
    for version in [NEW_VERSION, OLD_VERSION]
]

[
    java_import(
        name = "report-%s" % version,
        jars = ["org.jacoco.report-%s.jar" % version],
        srcjar = "org.jacoco.report-%s-sources.jar" % version,
        exports = [
            ":core-%s" % version,
            "//third_party:asm",
        ],
    )
    for version in [NEW_VERSION, OLD_VERSION]
]

[
    java_import(
        name = "blaze-agent-%s" % version,
        jars = ["jacocoagent-%s.jar" % version],
    )
    for version in [NEW_VERSION, OLD_VERSION]
]

[
    java_import(
        name = "blaze-agent-neverlink-%s" % version,
        jars = ["jacocoagent-%s.jar"  % version],
        neverlink = 1,
    )
    for version in [NEW_VERSION, OLD_VERSION]
]
