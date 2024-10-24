// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for dependencies between C++ libraries. */
@RunWith(JUnit4.class)
public final class CcBadDependenciesTest extends BuildViewTestCase {
  private ConfiguredTarget configure(String targetLabel) throws Exception {
    return getConfiguredTarget(targetLabel);
  }

  @Test
  public void testRejectsSingleUnknownSourceFile() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['unknown.oops'])");
    scratch.file("foo/unknown.oops", "foo");
    configure("//foo:foo");
    assertContainsEvent(
        getErrorMsgMisplacedFiles("srcs", "cc_library", "@@//foo:foo", "@@//foo:unknown.oops"));
  }

  @Test
  public void testAcceptsDependencyWithAtLeastOneGoodSource() throws Exception {
    scratch.file(
        "dependency/BUILD",
        """
        genrule(
            name = "goodandbad_gen",
            outs = [
                "good.cc",
                "bad.oops",
            ],
            cmd = "/bin/true",
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        cc_library(
            name = "foo",
            srcs = ["//dependency:goodandbad_gen"],
        )
        """);
    configure("//foo:foo");
  }

  @Test
  public void testRejectsBadGeneratedFile() throws Exception {
    setBuildLanguageOptions("--experimental_builtins_injection_override=+cc_library");
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "dependency/BUILD",
        """
        genrule(
            name = "generated",
            outs = ["bad.oops"],
            cmd = "/bin/true",
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        cc_library(
            name = "foo",
            srcs = ["//dependency:generated"],
        )
        """);
    configure("//foo:foo");
    assertContainsEvent(
        "attribute srcs: '@@//dependency:generated' does not produce any cc_library srcs files");
  }
}
