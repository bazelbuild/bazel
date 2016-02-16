# A target to ensure the servlet-api is not linked in the webapp.
java_library(
    name = "javax.servlet.api",
    neverlink = 1,
    visibility = ["//visibility:public"],
    exports = ["@javax_servlet_api//jar:jar"],
)

filegroup(
    name = "runner_template",
    srcs = ["appengine_runner.sh.template"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "deploy_template",
    srcs = ["appengine_deploy.sh.template"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)
