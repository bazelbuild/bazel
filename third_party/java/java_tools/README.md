# Tools required by the Bazel rules for Java.

To update all the Java tools please run from the workspace root:

```
./third_party/java/java_tools/update_java_tools.sh
```

To update individual Java tools please see the instructions for each tool
below.

## JavaBuilder

`JavaBuilder` is a tools used by Bazel to compile Java code. It is a wrapper
around `javac` and also provides:

* Strict Java dependency enforcement
* [Error prone](http://errorprone.info) checks
* Header Compilation
* Reduced Classpath Optimization

The code of JavaBuilder is under //src/java_tools/buildjar.

`JavaBuilder_deploy.jar` was built with bazel 0.21.0 from commit
[3256863e75f96c1ad9537b5b12f66a0e5b174781](https://github.com/bazelbuild/bazel/commit/3256863e75f96c1ad9537b5b12f66a0e5b174781)

```
bazel build //src/java_tools/buildjar:JavaBuilder_deploy.jar
cp -f bazel-bin/src/java_tools/buildjar/JavaBuilder_deploy.jar third_party/java/java_tools/JavaBuilder_deploy.jar
```

### VanillaJavaBuilder

VanillaJavaBuilder strips the additional features added in `JavaBuilder`. When enabled
Bazel will invoke `javac` directly.

`VanillaJavaBuilder_deploy.jar` was built with bazel 0.21.0 from commit
[3256863e75f96c1ad9537b5b12f66a0e5b174781](https://github.com/bazelbuild/bazel/commit/3256863e75f96c1ad9537b5b12f66a0e5b174781)

```
bazel build //src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar
cp -f bazel-bin/src/java_tools/buildjar/VanillaJavaBuilder_deploy.jar third_party/java/java_tools/VanillaJavaBuilder_deploy.jar
```

## GenClassJar

`GenClassJar` was built with bazel 0.21.0 from commit
[3256863e75f96c1ad9537b5b12f66a0e5b174781](https://github.com/bazelbuild/bazel/commit/3256863e75f96c1ad9537b5b12f66a0e5b174781)

```
bazel build //src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar
cp -f bazel-bin/src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass/GenClass_deploy.jar third_party/java/java_tools/GenClass_deploy.jar
```

## JUnit Runner

```
bazel build //src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar
cp -f bazel-bin/src/java_tools/junitrunner/java/com/google/testing/junit/runner/Runner_deploy.jar third_party/java/java_tools/Runner_deploy.jar
```