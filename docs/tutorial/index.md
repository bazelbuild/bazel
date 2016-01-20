---
layout: documentation
title: Tutorial - Introduction
---

# Tutorial - Introduction

You can use Bazel to build a variety of software outputs, including
Linux and Mac OS X applications written in Java, C++ and Objective-C. You can
also use Bazel to build software for other platforms or written in other
languages.

This tutorial shows how to use Bazel to build the following:

*   An Android app
*   An iOS app
*   A mobile backend server running on App Engine

In this tutorial, you'll learn how to:

*   Set up a Bazel workspace and create a `WORKSPACE` file
*   Create `BUILD` files that contain the instructions used by Bazel to build
    the software
*   Run builds using the Bazel command line tool

## Requirements

You can follow the steps in this tutorial on either a Linux or Mac OS X system.
However, you can only build the iOS app if you are running Bazel on OS X. If
you are using Linux, you can skip the iOS instructions and still complete
the rest of the tutorial steps.

## Sample project

You don't have to write your own mobile apps and backend server to use this
tutorial. Instead, you'll use a sample project hosted on GitHub. The sample
project is hosted at the following location:

[https://github.com/bazelbuild/examples/](https://github.com/bazelbuild/examples/)

You'll grab the sample project files in the next step in this tutorial.

## What's next

Let's start off by [setting up](environment.md) the tutorial environment.
