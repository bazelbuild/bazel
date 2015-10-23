# Scala Rules for Bazel

## Overview

This rule is used for building [Scala][scala] projects with Bazel. There are
currently two rules, `scala_library` and `scala_binary`. More features will be
added in the future, e.g. `scala_test`.

[scala]: http://www.scala-lang.org/

## scala_library

`scala_library` generates a `.jar` file from `.scala` source files.
In order to make a java rule use this jar file, use the `java_import` rule.

The current implementation assumes that the files `/usr/bin/scalac` and
`/usr/share/java/scala-library.jar` exist.

## scala_binary

`scala_binary` generates a Scala executable. It may depend on `scala_library`
and `java_library` rules.

A `scala_binary` requires a `main_class` attribute.
