// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import com.google.common.collect.ImmutableMap;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.skylark.skylint.Linter.FileFacade;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the lint done by {@link DeprecationChecker}. */
@RunWith(JUnit4.class)
public class DeprecationCheckerTest {
  private static List<Issue> findIssuesIgnoringDependencies(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return DeprecationChecker.check(
        Paths.get("/fake"), ast, _path -> content.getBytes(StandardCharsets.ISO_8859_1));
  }

  @Test
  public void symbolDeprecatedInSameFile() {
    String errorMessages =
        findIssuesIgnoringDependencies(
                "def f():",
                "  g()",
                "  h()",
                "  print(x)",
                "def g():",
                "  '''Foo.",
                "  ",
                "  Deprecated:",
                "    Reason.'''",
                "def h():",
                "  '''Bar.",
                "  ",
                "  This function is DEPRECATED for some reason.",
                "  The deprecation should really be documented in a 'Deprecated:' section",
                "  but the linter should recognize this kind of deprecation as well'''",
                "x = 0",
                "'''A deprecated variable.",
                "",
                "Deprecated:",
                "  Reason.",
                "'''")
            .toString();
    Truth.assertThat(errorMessages)
        .contains("2:3: usage of 'g' is deprecated: Reason. [deprecated-symbol]");
    Truth.assertThat(errorMessages)
        .contains(
            "3:3: usage of 'h' is deprecated: This function is DEPRECATED for some reason."
                + " [deprecated-symbol]");
    Truth.assertThat(errorMessages)
        .contains("4:9: usage of 'x' is deprecated: Reason. [deprecated-symbol]");
  }

  @Test
  public void deprecatedFunctionAliasing() {
    String errorMessages =
        findIssuesIgnoringDependencies(
                "def f():",
                "  h = g",
                "  h()",
                "def g():",
                "  '''Foo.",
                "  ",
                "  Deprecated:",
                "    reason'''")
            .toString();
    Truth.assertThat(errorMessages)
        .contains("2:7: usage of 'g' is deprecated: reason [deprecated-symbol]");
  }

  @Test
  public void functionNotDeprecated() {
    Truth.assertThat(
            findIssuesIgnoringDependencies(
                "def f():",
                "  g()",
                "def g():",
                "  '''This is a good function.",
                "",
                "  It is emphatically not deprecated.'''"))
        .isEmpty();
  }

  @Test
  public void noWarningsInsideDeprecatedFunctions() {
    Truth.assertThat(
            findIssuesIgnoringDependencies(
                "def f():",
                "  '''A deprecated function calling another deprecated function -> no warning",
                "",
                "  Deprecated:",
                "    This function is deprecated.",
                "  '''",
                "  g()",
                "",
                "def g():",
                "  '''Another deprecated function",
                "",
                "  Deprecated:",
                "    This function is deprecated.'''"))
        .isEmpty();
  }

  @Test
  public void shadowingWorks() {
    Truth.assertThat(
            findIssuesIgnoringDependencies(
                "def f():",
                "  bad = good",
                "  bad()",
                "def good(): pass",
                "def bad():",
                "  '''This is a deprecated function.",
                "",
                "  Deprecated:",
                "    reason'''"))
        .isEmpty();
  }

  private static final Map<String, String> files =
      ImmutableMap.<String, String>builder()
          .put("/WORKSPACE", "")
          .put("/pkg1/BUILD", "")
          .put(
              "/pkg1/foo.bzl",
              String.join(
                  "\n",
                  "load('//pkg2:bar.bzl', 'baz', foo = 'bar') # test aliasing",
                  "load(':qux.bzl', 'qux') # test package label",
                  "foo()",
                  "qux()"))
          .put(
              "/pkg1/qux.bzl",
              String.join(
                  "\n",
                  "load(':does_not_exist.bzl', 'foo') # test error handling",
                  "def qux():",
                  "  '''qux",
                  "",
                  "  Deprecated:",
                  "    qux is deprecated'''",
                  "return"))
          .put("/pkg2/BUILD", "")
          .put(
              "/pkg2/bar.bzl",
              String.join(
                  "\n",
                  "load('//pkg2/pkg3:baz.bzl', bar = 'baz') # test aliasing",
                  "bar()",
                  "def baz():",
                  "  '''baz",
                  "",
                  "  Deprecated:",
                  "    baz is deprecated",
                  "  '''"))
          .put("/pkg2/pkg3/BUILD", "")
          .put(
              "/pkg2/pkg3/baz.bzl",
              String.join(
                  "\n",
                  "def baz():",
                  "  '''baz",
                  "",
                  "  Deprecated:",
                  "    baz is deprecated",
                  "  '''"))
          .build();

  private static final FileFacade testFileFacade = DependencyAnalyzerTest.toFileFacade(files);

  private static List<Issue> findIssuesIncludingDependencies(String pathString) throws IOException {
    Path path = Paths.get(pathString);
    return DeprecationChecker.check(path, testFileFacade.readAst(path), testFileFacade);
  }

  @Test
  public void deprecatedFunctionsInDirectDependency() throws Exception {
    String errorMessages = findIssuesIncludingDependencies("/pkg1/foo.bzl").toString();
    Truth.assertThat(errorMessages)
        .contains(
            "4:1-4:3: usage of 'qux' (imported from /pkg1/qux.bzl)"
                + " is deprecated: qux is deprecated");
    errorMessages = findIssuesIncludingDependencies("/pkg2/bar.bzl").toString();
    Truth.assertThat(errorMessages)
        .contains(
            "2:1-2:3: usage of 'bar' (imported from /pkg2/pkg3/baz.bzl, named 'baz' there)"
                + " is deprecated: baz is deprecated");
  }

  @Test
  public void deprecatedFunctionsInTransitiveDependencies() throws Exception {
    String errorMessages = findIssuesIncludingDependencies("/pkg1/foo.bzl").toString();
    Truth.assertThat(errorMessages)
        .contains(
            "3:1-3:3: usage of 'foo' (imported from /pkg2/pkg3/baz.bzl, named 'baz' there)"
                + " is deprecated: baz is deprecated");
  }
}
