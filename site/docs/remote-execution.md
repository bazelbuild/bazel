---
layout: documentation
title: Remote Execution Overview
---

# Remote Execution Overview

By default, Bazel executes builds and tests on your local machine. Remote
execution of a Bazel build allows you to distribute build and test actions
across multiple machines, such as a datacenter.

Remote execution provides the following benefits:

*  Faster build and test execution through scaling of nodes available
   for parallel actions
*  A consistent execution environment for a development team
*  Reuse of build outputs across a development team

Bazel uses an open-source
[gRPC protocol](https://github.com/bazelbuild/remote-apis)
to allow for remote execution and remote caching.

## Remote Execution Services

To run Bazel with remote execution, you can use one of the following:

<!-- to-do: When we have a public post to link to, include: *  Use [Cloud Build for Bazel](), which is a remote execution service from Google -->

*  Manual
    *   Use the
        [gRPC protocol](https://github.com/bazelbuild/remote-apis)
        directly to create your own remote execution service.
*   Self-hosted
    *   [Buildbarn](https://github.com/buildbarn)
    *   [Buildfarm](https://github.com/bazelbuild/bazel-buildfarm)
    *   [BuildGrid](https://gitlab.com/BuildGrid/buildgrid)
*   Hosted
    *   Remote Build Execution, which is a remote execution service from Google.
        Joining the
        [RBE Alpha Customers group](https://groups.google.com/forum/#!forum/rbe-alpha-customers)
        will give you full access to the official documentation.
        To begin using the service, fill out this
        [short information form](https://docs.google.com/forms/d/e/1FAIpQLScBai-iQ2tn7RcGcsz3Twjr4yDOeHowrb6-3v5qlgS69GcxbA/viewform).

## Requirements for Remote Execution

Remote execution of Bazel builds imposes a set of mandatory configuration
constraints on the build. For more information, see
[Adapting Bazel Rules for Remote Execution](remote-execution-rules.html).
