load("/tools/build_rules/appengine/appengine", "appengine_war")

appengine_war(
    name = "dash",
    data = ["//src/tools/dash/src/main/webapp"],
    data_path = "/src/tools/dash/src/main/webapp",
    jars = ["//src/tools/dash/src/main/java:java-bin_deploy.jar"],
)
