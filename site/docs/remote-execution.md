---
layout: documentation
title: Remote execution overview
---

# Remote Execution Overview

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
    *   [BuildBuddy](https://www.buildbuddy.io) -- Remote build execution,
        caching, and results UI.
    *   [Flare](https://www.flare.build) --  Providing a cache + CDN for Bazel
        artifacts and Apple-focused remote builds in addition to build & test analytics.

## Requirements

Remote execution of Bazel builds imposes a set of mandatory configuration
constraints on the build. For more information, see
[Adapting Bazel Rules for Remote Execution](remote-execution-rules.html).
