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

import com.google.common.truth.Truth;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the lint done by {@link LoadStatementChecker}. */
@RunWith(JUnit4.class)
public class LoadStatementCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return LoadStatementChecker.check(ast);
  }

  @Test
  public void loadStatementsAtTheTop() {
    Truth.assertThat(
            findIssues(
                "'''Docstring'''",
                "load(':foo.bzl', 'foo')",
                "load(':bar.bzl', 'bar')",
                "foo = 'bar'"))
        .isEmpty();
    Truth.assertThat(findIssues("load(':foo.bzl', 'foo')", "foo = 'bar'")).isEmpty();
    Truth.assertThat(findIssues("")).isEmpty();
    Truth.assertThat(findIssues("'''Docstring'''")).isEmpty();
  }

  @Test
  public void loadStatementNotAtTheTop() {
    String errorMessage =
        findIssues(
                "'''Docstring'''",
                "print('This statement should be after the load.')",
                "load(':foo.bzl', 'foo')")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(
            ":3:1: load statement should be at the top of the file (after the docstring)");
    errorMessage =
        findIssues(
                "'''Docstring'''",
                "load(':foo.bzl', 'foo')",
                "print('foo')",
                "load(':bar.bzl', 'bar')")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(
            ":4:1: load statement should be at the top of the file (after the docstring)");
  }
}
