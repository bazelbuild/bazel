// Copyright 2007 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.test;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit test for {@link TestSuite}.
 */
@RunWith(JUnit4.class)
public class TestSuiteConfiguredTargetTest extends BuildViewTestCase {

  // Regression test for bug #878589, "CRASH: Blaze omits implied dependencies
  // of test_suite during loading phase".  The loading phase was failing to
  // follow the implicit edges from (some) test_suites to each *test rule in the
  // same package.
  @Test
  public void testLoadingOfImplicitTestSuiteTests() throws Exception {
    checkError(
        "x",
        "x",
        "no such package 'no/such/package': BUILD file not found",
        "test_suite(name='x')",
        "load('@rules_cc//cc:cc_test.bzl', 'cc_test')",
        "cc_test(name='sometest', data=['//no/such/package'])");
  }

  // Regression test for bug #18483315: "implicit tests in test_suite broken [release blocker]"
  @Test
  public void testHasRunfilesForImplicitTests() throws Exception {
    scratch.file(
        "c/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")

        foo_test(
            name = "t",
            srcs = ["t.sh"],
            data = ["data"],
        )

        test_suite(name = "s")
        """);
    ConfiguredTarget suite = getConfiguredTarget("//c:s");
    Runfiles runfiles = suite.getProvider(RunfilesProvider.class).getDataRunfiles();
    assertThat(ActionsTestUtil.baseArtifactNames(runfiles.getArtifacts())).contains("data");
  }

  // Regression test for bug #880514, "CRASH: Blaze crashes when a
  // test--test_suite cycle is encountered".
  @Test
  public void testTestTestSuiteCycleDoesntCrash() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "x/BUILD",
        """
        py_test(
            name = "y",
            srcs = ["y.py"],
            data = [":x"],
        )

        test_suite(name = "x")
        """);
    getConfiguredTarget("//x:y");
  }

  // TODO(ulfjack): This was a mistake - test suites shouldn't have runfiles.
  @Test
  public void testRunfiles() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("//test_defs:foo_binary.bzl", "foo_binary")
        load("//test_defs:foo_test.bzl", "foo_test")

        foo_test(
            name = "test",
            srcs = ["test.sh"],
            data = ["test.txt"],
        )

        test_suite(
            name = "suite",
            tests = [":test"],
        )

        foo_binary(
            name = "shbin",
            testonly = 1,
            srcs = ["shbin.sh"],
            data = [":suite"],
        )
        """);
    Set<String> runfilesNames =
        artifactsToStrings(
            getRunfilesSupport("//foo:shbin").getRunfilesTree().getMapping().values());
    assertThat(runfilesNames)
        .containsExactly(
            "bin foo/shbin",
            "src foo/shbin.sh",
            "bin foo/test",
            "src foo/test.sh",
            "src foo/test.txt");
  }

  @Test
  public void testMakeSureDepsDoesNotExist() throws Exception {
    checkError("broken", "suite", "no such attribute 'deps'",
        "test_suite(name = 'suite', deps = [])");
  }

  @Test
  public void testMakeSureDataDoesNotExist() throws Exception {
    checkError("broken", "suite", "no such attribute 'data'",
        "test_suite(name = 'suite', data = [])");
  }

  /**
   * Regression test for b/12446370: dependencies on test_suite rules no longer have tag filtering
   * applied
   *
   * <p>Filtered tests should not show up in the runfiles.
   */
  @Test
  public void testRunfilesDoesNotContainFilteredTest() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")

        foo_test(
            name = "test",
            srcs = ["test.sh"],
            data = ["test.txt"],
        )

        test_suite(
            name = "suite",
            tags = ["required"],
        )
        """);
    ConfiguredTarget target = getConfiguredTarget("//foo:suite");
    RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);
    assertThat(runfilesProvider.getDataRunfiles().getAllArtifacts().toList()).isEmpty();
  }
}
