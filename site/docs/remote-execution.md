---
layout: documentation
title: Remote execution overview
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/docs/remote-execution" style="color: #0000EE;">https://bazel.build/docs/remote-execution</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


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

For a list of commercially supported remote execution services as well as
self-service tools, see
[Remote Execution Services](https://www.bazel.build/remote-execution-services.html)

## Requirements

Remote execution of Bazel builds imposes a set of mandatory configuration
constraints on the build. For more information, see
[Adapting Bazel Rules for Remote Execution](remote-execution-rules.html).
