# Description:
#   JaCoCo is a free code coverage library for Java, created by the EclEmma team.
#
# https://search.maven.org/remotecontent?filepath=org/jacoco/jacoco/0.8.11/jacoco-0.8.11.zip

load("//src:release_archive.bzl", "release_archive")

licenses(["reciprocal"])  # EPL 1.0 (Eclipse Public License)

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

LASTVERSION = "0.8.11"

VERSIONS = ["0.8.11"]

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

release_archive(
    name = "jacoco_jars_zip",
    srcs = ["LICENSE", ":source_jars"] + [
        ":agent",
        ":blaze-agent",
        ":core",
        ":report",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
    package_dir = "java_tools/third_party/java/jacoco",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "source_jars",
    srcs = [":copy_source_jars"],
)

load(":extract_source_jars.bzl", "extract_source_jars")

extract_source_jars(
    name = "agent_source_jar",
    deps = ["@maven//:org_jacoco_org_jacoco_agent"],
)

extract_source_jars(
    name = "core_source_jar",
    deps = ["@maven//:org_jacoco_org_jacoco_core"],
)

extract_source_jars(
    name = "report_source_jar",
    deps = ["@maven//:org_jacoco_org_jacoco_report"],
)

genrule(
    name = "copy_source_jars",
    srcs = [
        ":agent_source_jar",
        ":core_source_jar",
        ":report_source_jar",
    ],
    outs = [
        "org.jacoco.agent-%s-sources.jar" % LASTVERSION,
        "org.jacoco.core-%s-sources.jar" % LASTVERSION,
        "org.jacoco.report-%s-sources.jar" % LASTVERSION,
    ],
    cmd = "cp $(location :agent_source_jar) $(location org.jacoco.agent-%s-sources.jar) && " % LASTVERSION +
          "cp $(location :core_source_jar) $(location org.jacoco.core-%s-sources.jar) && " % LASTVERSION +
          "cp $(location :report_source_jar) $(location org.jacoco.report-%s-sources.jar)" % LASTVERSION,
)

filegroup(
    name = "transitive_sources",
    srcs = [
        "LICENSE",
        ":source_jars",
        "//third_party:asm_source_jars",
    ],
)

alias(
    name = "agent",
    actual = "@maven//:org_jacoco_org_jacoco_agent",
)

alias(
    name = "core",
    actual = "@maven//:org_jacoco_org_jacoco_core",
)

alias(
    name = "report",
    actual = "@maven//:org_jacoco_org_jacoco_report",
)

alias(
    name = "blaze-agent",
    actual = "@maven//:org_jacoco_org_jacoco_agent_runtime",
)
