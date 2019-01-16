# Tools required by the Bazel rules for Java.

To see how to update the tools run:

```
third_party/java/java_tools/update_java_tools.sh help
```

The following tools were built with bazel 0.21.0 at commit 252c82cd135f1bcdadfc961c127895e9b2347b77

third_party/java/java_tools/JavaBuilder_deploy.jar
third_party/java/java_tools/VanillaJavaBuilder_deploy.jar
third_party/java/java_tools/GenClass_deploy.jar
third_party/java/java_tools/Runner_deploy.jar
third_party/java/java_tools/ExperimentalRunner_deploy.jar
third_party/java/java_tools/JacocoCoverage_jarjar_deploy.jar
third_party/java/java_tools/turbine_deploy.jar
third_party/java/java_tools/turbine_direct_binary_deploy.jar
third_party/java/java_tools/SingleJar_deploy.jar

The following tools were built with bazel 0.21.0 at commit 7f97ed4df69c2d9358a063cc7bb310c615ff6496 by running:
$ third_party/java/java_tools/update_java_tools.sh ImportDepsChecker

third_party/java/java_tools/ImportDepsChecker_deploy.jar