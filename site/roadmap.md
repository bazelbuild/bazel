---
layout: contribute
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
      <td rowspan="9"><b>Alpha</b><br/><span class="label label-default">Released</span></td>
      <td rowspan="9"><b>2015&#8209;03&#8209;24</b></td>
      <td>Linux &amp; OS X Support</td>
    </tr>
    <tr><td>C++ (<a href="http://bazel.build/docs/be/c-cpp.html#cc_binary">build</a> and <a href="http://bazel.build/docs/be/c-cpp.html#cc_test">test</a>)</td></tr>
    <tr><td>Java (<a href="http://bazel.build/docs/be/java.html#java_binary">build</a> and <a href="http://bazel.build/docs/be/java.html#java_test">test</a>)</td></tr>
    <tr><td>Objective-C for iOS (<a href="http://bazel.build/docs/be/objective-c.html#objc_binary">build</a>)</td></tr>
    <tr><td>Python (<a href="http://bazel.build/docs/be/python.html#py_binary">build</a>)</td></tr>
    <tr><td>iOS applications (<a href="http://bazel.build/docs/be/objective-c.html#ios_application">build</a>)</td></tr>
    <tr><td>Skylark extension mechanism (<a href="http://bazel.build/docs/skylark/index.html">build</a>)</td></tr>
    <tr><td>Basic test suite on GitHub</td></tr>
    <tr><td>Support for fetching dependencies from <a href="http://bazel.build/docs/be/workspace.html#maven_jar">Maven repositories</a>
        and <a href="http://bazel.build/docs/be/workspace.html#http_archive">web servers</a></td></tr>
    <tr>
      <td rowspan="44"><b><a name="beta"></a>Beta</b></td>
      <td rowspan="9">
         <a href="https://github.com/bazelbuild/bazel/releases/tag/0.1.0"><b>0.1</b>
         <br/><span class="label label-default">Released</span></a>
      </td>
      <td rowspan="9"><b>2015&#8209;09&#8209;01</b></td>
      <td>P0. Binary distribution for Linux & OS X</td<
    </tr>
    <tr><td>P0. Public <a href="http://ci.bazel.build">continuous integration system</a></td></tr>
    <tr><td>P0. Support for <a href="http://bazel.build/docs/external.html">fetching transitive dependencies from Maven Central</a></td></tr>
    <tr><td>P0. Android application (<a href="http://bazel.build/docs/be/android.html#android_binary">build</a>
        and <a href="http://bazel.build/docs/bazel-user-manual.html#mobile-install">install</a>)</td></tr>
    <tr><td>P1. Support for <a href="http://bazel.build/docs/external.html">prefetching and caching remote dependencies</a></td></tr>
    <tr><td>P1. Docker (<a href="http://bazel.build/docs/be/docker.html">build and load</a>)</td></tr>
    <tr><td>P2. <a href="http://bazel.build/docs/bazel-user-manual.html#sandboxing">Sandboxing of actions for Linux</a></td></tr>
    <tr><td>P2. AppEngine (<a href="http://bazel.build/docs/be/appengine.html">build and load</a>)</td></tr>
    <tr><td>P2. <a href="http://bazel.build/blog/2015/07/29/dashboard-dogfood.html">Test result dashboard</a></tr></td>
    <tr>
      <td rowspan="5">
        <a href="https://github.com/bazelbuild/bazel/releases/tag/0.2.0"><b>0.2</b>
        <br/><span class="label label-default">Released</span></a>
      </td>
      <td rowspan="5"><b>2016&#8209;02&#8209;18</b></td>
      <td>P0. <a href="https://github.com/bazelbuild/bazel/tree/master/src/test/java/com/google/devtools">Significantly increase test coverage</a></td>
    </tr>
    <tr><td>P0. Support for fetching <a href="http://bazel.build/docs/external.html">remote</a> <a href="http://bazel.build/docs/be/functions.html#load">Skylark rules</a></td></tr>
    <tr><td>P2. <a href="https://github.com/bazelbuild/rules_go">Go language support (build and tests)</a></td></tr>
    <tr><td>P2. <a href="https://github.com/bazelbuild/bazel/releases/latest">Debian packages for Bazel</a></td></tr>
    <tr><td>P2. <a href="http://braumeister.org/formula/bazel">OS X homebrew recipe for distributing Bazel</a></td></tr>
    <tr>
      <td rowspan="5">
        <a href="https://github.com/bazelbuild/bazel/releases/tag/0.3.0"><b>0.3</b>
        <br/><span class="label label-default">Released</span></a>
      </td>
      <td rowspan="5"><b>2016&#8209;06&#8209;10</b></td>
      <td>P0. <a href="http://bazel.build/docs/windows.html">Bazel can bootstrap itself on Windows without requiring admin privileges</a></td></tr>
    </tr>
    <tr><td>P1. <a href="http://bazel.build/blog/2016/06/10/ide-support.html">Interface for IDE support</a></td></tr>
    <tr><td>P1. IDE support for <a href="http://tulsi.bazel.build">Xcode (stable)</a> and <a href="https://github.com/bazelbuild/e4b">Eclipse (experimental)</a></td></tr>
    <tr><td>P1. <a href="https://docs.google.com/document/d/1jKbNXOVp2T1zJD_iRnVr8k5D0xZKgO8blMVDlXOksJg">Custom remote repositories using Skylark</a></td></tr>
    <tr><td>P2. <a href="https://github.com/bazelbuild/bazel/commit/79adf59e2973754c8c0415fcab45cd58c7c34697">Prototype for distributed caching of build artifact</a></td></tr>
    <tr>
      <td rowspan="2">
        <a href="https://github.com/bazelbuild/bazel/releases/tag/0.4.0"><b>0.4</b>
        <br/><span class="label label-default">Released</span></a>
      </td>
      <td rowspan="2"><b>2016&#8209;11&#8209;02</b></td>
      <td>P0. <a href="https://github.com/bazelbuild/bazel/commit/490f250b27183a886cf70a5fe9e99d9428141b34">Persistent Java compiler is enabled</a></td>
    </tr>
    <tr>
      <td>P2. <a href="https://github.com/bazelbuild/bazel/commit/7b825b8ea442246aabfa6a5a8962abd70855d0da">Sandboxing of action for OS X</a></td>
    </tr>
    <tr>
      <td rowspan="5"><b>0.5</b></td>
      <td rowspan="5"><b>2017&#8209;03</b></td>
      <td>P0. Support for building and testing Java, C++ and Python on Windows</td>
    </tr>
    <tr><td>P1. Initial API for a Build Event Protocol</td></tr>
    <tr><td>P1. Support for coverage for Java</td></tr>
    <tr><td>P1. Bazel installer optionally bundles the JDK</td></tr>
    <tr><td>P2. Repository rules no longer have invalidation issues</td></tr>
    <tr>
      <td rowspan="4"><b>0.6</b></td>
      <td rowspan="4"><b>2017&#8209;06</b></td>
      <td>P0. Stable API for Remote execution including platform description</td>
    </tr>
    <tr><td>P0. List of feature to deprecate until version 1.0 are tracked in a publicly available document</td></tr>
    <tr><td>P1. Bazel on Windows does not need to install MSYS</td></tr>
    <tr><td>P2. Bazel can load workspace recursively</td></tr>
    <tr>
      <td rowspan="8"><b>0.7</b></td>
      <td rowspan="8"><b>2017&#8209;09</b></td>
      <td>P0. Skylark is fully documented; strategy for user-provided Skylark rule documentation</td>
    </tr>
    <tr><td>P0. Support for Android integration testing</td></tr>
    <tr><td>P0. Support for Robolectric test for Android</td></tr>
    <tr><td>P1. The Build Event Protocol is stable</td></tr>
    <tr><td>P1. Support for coverage can be extended in Skylark and C++</td></tr>
    <tr><td>P1. Support for testing Skylark rules</td></tr>
    <tr><td>P2. All external repositories can use the local cache</td></tr>
    <tr><td>P2. Local caching of build artifacts</td></tr>
    <tr>
      <td rowspan="4"><b>0.8</b></td>
      <td rowspan="4"><b>2017&#8209;12</b></td>
      <td>P0. Support for iOS integration testing</td>
    </tr>
    <tr><td>P1. Bazel can build Android application on Windows</td></tr>
    <tr><td>P1. Local caching of external repository is turned on by default and has a deletion strategy</td></tr>
    <tr><td>P1. Access to native rules functionality from Skylark (<a href="https://bazel.build/designs/2016/08/04/extensibility-for-native-rules.html">"sandwich"</a>)</td></tr>
   <tr>
      <td rowspan="2"><b>0.9</b></td>
      <td rowspan="2"><b>2018&#8209;03</b></td>
      <td>P0. Full Windows support</td>
    </tr>
    <tr><td>P1. Full test suite is open-sourced</td></tr>
    <tr>
      <td rowspan="7"><b><a name="stable"></a>Stable</b></td>
      <td rowspan="7"><b>1.0</b></td>
      <td rowspan="7"><b>2018&#8209;06</b></td>
      <td>P0. APIs are versioned</td>
    </tr>
    <tr><td>P0. Github is primary</td></tr>
    <tr><td>P1. Deprecated features are removed</td></tr>
    <tr><td>P1. Support policy is defined regarding LTS release</td></tr>
    <tr><td>P1. Public review process for design documents</td></tr>
    <tr><td>P1. Bazel respects the standard for Debian packaging</td></tr>
    <tr><td>P2. Bazel is in the list of debian package for the next stable</td></tr>
  </tbody>
</table>

