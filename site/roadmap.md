---
layout: community
title: Roadmap
---

# Bazel Feature Roadmap

This document describes the Bazel team's plans for introducing features that
will be incorporated into version 1.0. Note that this roadmap only includes
features that the Bazel team itself intends to support. We anticipate that a
number of other features will be added by code contributors.

For the alpha and beta releases, the Bazel team will maintain two code
repositories:

*   A Google-internal repository, containing both Bazel code and
    Google-specific extensions and features
*   An external [GitHub repository](https://github.com/bazelbuild/bazel),
    containing only the Bazel code.

We anticipate making the external repository *primary* in the future, that is,
code from Google and non-Google contributors will be committed and tested in the
external repository first, then imported into the internal repository. For
the alpha and beta releases, however, the internal repository will be primary.
Changes to Bazel code will be frequently pushed from the internal to
the external repository.

## Feature list

In the following table, each feature is associated with a corresponding
milestone. The convention for the priorities are:

*   P0 feature will block the milestone; we will delay the milestone date
    until the feature is shipped.
*   P1 feature can delay the milestone if the feature can be shipped with a
    reasonable delay (2 months max).
*   P2 feature will be dropped and rescheduled for later rather than delaying
    the milestone.

We will update this list when reaching each milestone; some milestones may also
be refined if appropriate.

<table class="table table-condensed table-bordered">
  <colgroup>
    <col class="roadmap-col-phase"/>
    <col class="roadmap-col-milestone"/>
    <col class="roadmap-col-date"/>
    <col class="roadmap-col-features"/>
  </colgroup>
  <thead>
    <tr>
      <th>Phase</th>
      <th>Milestone</th>
      <th>Target date</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9"><b><a name="alpha"></a>Alpha</b></td>
      <td rowspan="9"><b>Alpha</b></td>
      <td rowspan="9"><b>2015&#8209;03</b></td>
      <td>Linux &amp; OS X Support</td>
    </tr>
    <tr><td>C++ (<a href="http://bazel.io/docs/be/c-cpp.html#cc_binary">build</a> and <a href="http://bazel.io/docs/be/c-cpp.html#cc_test">test</a>)</td></tr>
    <tr><td>Java (<a href="http://bazel.io/docs/be/java.html#java_binary">build</a> and <a href="http://bazel.io/docs/be/java.html#java_test">test</a>)</td></tr>
    <tr><td>Objective-C for iOS (<a href="http://bazel.io/docs/be/objective-c.html#objc_binary">build</a>)</td></tr>
    <tr><td>Python (<a href="http://bazel.io/docs/be/python.html#py_binary">build</a>)</td></tr>
    <tr><td>iOS applications (<a href="http://bazel.io/docs/be/objective-c.html#ios_application">build</a>)</td></tr>
    <tr><td>Skylark extension mechanism (<a href="http://bazel.io/docs/skylark/index.html">build</a>)</td></tr>
    <tr><td>Basic test suite on GitHub</td></tr>
    <tr><td>Support for fetching dependencies from <a href="http://bazel.io/docs/be/workspace.html#maven_jar">Maven repositories</a>
        and <a href="http://bazel.io/docs/be/workspace.html#http_archive">web servers</a></td></tr>
    <tr>
      <td rowspan="19"><b><a name="beta"></a>Beta</b></td>
      <td rowspan="9"><b>0.1</b></td>
      <td rowspan="9"><b>2015&#8209;09</b></td>
      <td>P0. Binary distribution for Linux & OS X</td<
    </tr>
    <tr><td>P0. Public <a href="http://ci.bazel.io">continuous integration system</a></td></tr>
    <tr><td>P0. Support for <a href="http://bazel.io/docs/external.html">fetching transitive dependencies from Maven Central</a></td></tr>
    <tr><td>P0. Android application (<a href="http://bazel.io/docs/be/android.html#android_binary">build</a>
        and <a href="http://bazel.io/docs/bazel-user-manual.html#mobile-install">install</a>)</td></tr>
    <tr><td>P1. Support for <a href="http://bazel.io/docs/external.html">prefetching and caching remote dependencies</a></td></tr>
    <tr><td>P1. Docker (<a href="http://bazel.io/docs/be/docker.html">build and load</a>)</td></tr>
    <tr><td>P2. <a href="http://bazel.io/docs/bazel-user-manual.html#sandboxing">Sandboxing of actions for Linux</a></td></tr>
    <tr><td>P2. AppEngine (<a href="http://bazel.io/docs/be/appengine.html">build and load</a>)</td></tr>
    <tr><td>P2. <a href="http://bazel.io/blog/2015/07/29/dashboard-dogfood.html">Test result dashboard</a></tr></td>
    <tr>
      <td rowspan="3"><b>0.2</b></td>
      <td rowspan="3"><b>2015&#8209;12</b></td>
      <td>P0. Significantly increase test coverage</td>
    </tr>
    <tr><td>P0. Support for fetching remote Skylark rules</td></tr>
    <tr><td>P2. Go language support (build and tests)</td></tr>
    <tr>
      <td rowspan="2"><b>0.3</b></td>
      <td rowspan="2"><b>2016&#8209;02</b></td>
      <td>P0. Bazel can bootstrap itself on Windows without requiring admin privileges</td></tr>
    </tr>
    <tr><td>P1. Interface for IDE support</td></tr>
    <tr>
      <td rowspan="3"><b>0.4</b></td>
      <td rowspan="3"><b>2016&#8209;04</b></td>
      <td>P0. Persistent Java compiler is enabled</td>
    </tr>
    <tr><td>P1. <a href="https://docs.google.com/document/d/1jKbNXOVp2T1zJD_iRnVr8k5D0xZKgO8blMVDlXOksJg">Custom remote repositories using Skylark</a></td></tr>
    <tr><td>P2. Sandboxing of action for OS X</td></tr>
    <tr>
      <td rowspan="2"><b>0.5</b></td>
      <td rowspan="2"><b>2016&#8209;06</b></td>
      <td>P0. Support for testing Android apps</td>
    </tr>
    <tr><td>P1. Distributed caching of build artifacts</td></tr>
    <tr>
      <td rowspan="11"><b><a name="stable"></a>Stable</b></td>
      <td rowspan="11"><b>1.0</b></td>
      <td rowspan="11"><b>2016&#8209;12</b></td>
      <td>P0. Extension APIs are stable and versioned</td>
    </tr>
    <tr><td>P0. Github repository is primary</td></tr>
    <tr><td>P0. Full Windows support for Android: Android feature set is identical for Windows and Linux/OS X</td></tr>
    <tr><td>P0. Android Studio interoperability</td></tr>
    <tr><td>P0. Support for testing iOS apps</td></tr>
    <tr><td>P1. Online repository of Skylark rules</td></tr>
    <tr><td>P2. Native protobuf support</td></tr>
    <tr><td>P2. Support testing using Google <a href="https://developers.google.com/cloud-test-lab/">Cloud Test Lab</a></td></tr>
    <tr><td>P2. Debian packages for Bazel</td></tr>
    <tr><td>P2. OS X homebrew recipe for distributing Bazel</td></tr>
    <tr><td>P2. Reference ("pull") remote docker images as an input to the build process</td></tr>
  </tbody>
</table>
