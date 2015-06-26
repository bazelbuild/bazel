---
layout: posts
title: Checking your Java errors with Error Prone.
---

We recently open-sourced our support for [Error Prone](http://errorprone.info).
[Error Prone](http://errorprone.info) checks for common mistakes in Java code
that will not be caught by the compiler.

We turned [Error Prone](http://errorprone.info) on by default but you can
easily turn it off by using the Javac option `--extra_checks:off`. To do so,
simply specify `--javacopts='-extra_checks:off'` to the list of Bazel's options.
You can also tune the checks error-prone will perform by using the
[`-Xep:` flags](http://errorprone.info/docs/flags).

See the [documentation of Error Prone](http://errorprone.info/docs) for more
on Error Prone.
