# Description:
#   JaCoCo is a free code coverage library for Java, created by the EclEmma team.

licenses(["reciprocal"])  # EPL 1.0 (Eclipse Public License)

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "agent",
    jars = ["org.jacoco.agent-0.7.5.201505241946.jar"],
    srcjar = "org.jacoco.agent-0.7.5.201505241946-src.jar",
)

java_import(
    name = "core",
    jars = ["org.jacoco.core-0.7.5.201505241946.jar"],
    srcjar = "org.jacoco.core-0.7.5.201505241946-src.jar",
    exports = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
    ],
)

filegroup(
    name = "core-jars",
    srcs = ["org.jacoco.core-0.7.5.201505241946.jar"],
)

java_import(
    name = "report",
    jars = ["org.jacoco.report-0.7.5.201505241946.jar"],
    srcjar = "org.jacoco.report-0.7.5.201505241946-src.jar",
    exports = [
        ":core",
        "//third_party:asm",
    ],
)

java_import(
    name = "blaze-agent",
    jars = ["jacocoagent.jar"],
)

java_import(
    name = "blaze-agent-neverlink",
    jars = ["jacocoagent.jar"],
    neverlink = 1,
)
