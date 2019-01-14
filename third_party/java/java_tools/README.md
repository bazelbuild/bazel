# Tools required by the Bazel rules for Java.

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
bazel build src/java_tools/buildjar:JavaBuilderDeploy
cp bazel-bin/src/java_tools/buildjar/JavaBuilder_deploy.jar third_party/java/java_tools/JavaBuilder_deploy.jar
```

