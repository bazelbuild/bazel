Project: /_project.yaml
Book: /_book.yaml

# Remote Execution Overview

{% include "_buttons.html" %}

This page covers the benefits, requirements, and options for running Bazel
with remote execution.

By default, Bazel executes builds and tests on your local machine. Remote
execution of a Bazel build allows you to distribute build and test actions
across multiple machines, such as a datacenter.

Remote execution provides the following benefits:

*  Faster build and test execution through scaling of nodes available
   for parallel actions
*  A consistent execution environment for a development team
*  Reuse of build outputs across a development team

Bazel uses an open-source
[gRPC protocol](https://github.com/bazelbuild/remote-apis){: .external}
to allow for remote execution and remote caching.

For a list of commercially supported remote execution services as well as
self-service tools, see
[Remote Execution Services](https://www.bazel.build/remote-execution-services.html){: .external}

## Requirements {:#requirements}

Remote execution of Bazel builds imposes a set of mandatory configuration
constraints on the build. For more information, see
[Adapting Bazel Rules for Remote Execution](/remote/rules).
