# Tools required by the Bazel rules for Java.

## JavaBuilder

`JavaBuilder` is a tools used by Bazel to compile Java code. It is a wrapper
around `javac` and also provides:

* Strict Java dependency enforcement
* [Error prone](http://errorprone.info) checks
* Header Compilation
* Reduced Classpath Optimization

The code of JavaBuilder is under //src/java_tools/buildjar.
`JavaBuilder_deploy.jar` was built at mainline 3ebdda5e0ebaf9dcce07b7232581fade07ac38fd.
