# Scala Rules for Bazel

## Overview

This rule is used for building [Scala][scala] projects with Bazel. There is
currently only one rule, `scala_library`. More features will be added in the
future, e.g. `scala_binary`, `scala_test`, etc.

[scala]: http://www.scala-lang.org/

### `scala_library`

`scala_library` generates a `.jar` file from `.scala` source files. In order to
make a java rule use this jar file, use the `java_import` rule.

The current implementation assumes that the files `/usr/bin/scalac` and
`/usr/share/java/scala-library.jar` exist.
