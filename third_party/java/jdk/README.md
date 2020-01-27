# Java compilers in Bazel

Bazel compiles Java code using a custom builder. This builder is called
JavaBuilder and its code lies in //src/java_tools/buildjar. To build Java
code, JavaBuilder use the Java compiler from the JDK. To support
[ErrorProne](http://errorprone.info) checks, we vendor a custom build
of the Java compiler code. This is the raw version of the Java compiler
from [OpenJDK](https://openjdk.java.net) but compiled for a lower
version of the JRE. Those builds are vendored in
//third_party/java/jdk/langtools.
