---
layout: documentation
title: Remote execution overview
---

# Remote execution overview

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

## Remote execution services

To run Bazel with remote execution, you can use one of the following:

*   Manual
    *   Use the
        [gRPC protocol](https://github.com/bazelbuild/remote-apis)
        directly to create your own remote execution service.
*   Self-service
    *   [Buildbarn](https://github.com/buildbarn)
    *   [Buildfarm](https://github.com/bazelbuild/bazel-buildfarm)
    *   [BuildGrid](https://gitlab.com/BuildGrid/buildgrid)
    *   [Scoot](https://github.com/twitter/scoot)
*   Commercial
    *   [EngFlow Remote Execution](https://www.engflow.com) -- Remote execution
        and remote caching service. Can be self-hosted or hosted.

## Requirements

Remote execution of Bazel builds imposes a set of mandatory configuration
constraints on the build. For more information, see
[Adapting Bazel Rules for Remote Execution](remote-execution-rules.html).
