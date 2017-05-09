# Java compilers in Bazel

Bazel compiles Java code using a custom builder. This builder is called
JavaBuilder and its code lies in //src/java_tools/buildjar. To build Java
code, JavaBuilder use the Java compiler from the JDK. To support
[ErrorProne](http://errorprone.info) checks, we vendor a custom build
of the Java compiler code. This is the raw version of the Java compiler
from [OpenJDK](https://openjdk.java.net) but compiled for a lower
version of the JRE. Those builds are vendored in
//third_party/java/jdk/langtools.

Currently Bazel supports running on a JRE 8 only because the default Java
compiler used (//third_party/java/jdk/langtools/javac-9-dev-r4023-2.jar) is the
Java compiler of OpenJDK 9 compiled to run on a JRE 8. This cannot
be built to run on a JRE 7 because of code incompatibility. Bazel's
JavaBuilder at HEAD cannot be linked with earlier version of the
Java compiler (it depends on some internals of the Java compiler).

To build a version of Bazel that can run on a JRE 7, we need to rely
on the version of JavaBuilder provided with Bazel 0.1.0
(//third_party/java/jdk/javabuilder/JavaBuilder_0.1.0_deploy.jar) which works
with a Java compiler of OpenJDK 8 compiled to run on a JRE 7
(//third_party/java/jdk/langtools/javac7.jar).
