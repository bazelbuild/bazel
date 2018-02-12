// Copyright 2018 The Bazel Authors. All rights reserved.
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

/**
 * Tests the lint done by {@link com.google.devtools.skylark.skylint.NativeRecursiveGlobChecker}.
 */
@RunWith(JUnit4.class)
public class NativeRecursiveGlobCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return NativeRecursiveGlobChecker.check(ast);
  }

  @Test
  public void testGoodJavaGlob() throws Exception {
    List<Issue> issues =
        findIssues("java_library(", "  name = \"foo\",", "  srcs = glob([\"*.java\"]),", ")");
    Truth.assertThat(issues).isEmpty();
  }

  @Test
  public void testCCGlob() throws Exception {
    List<Issue> issues =
        findIssues("cc_library(", "  name = \"foo\",", "  srcs = glob([\"**/*.cc\"]),", ")");
    // We don't currently test for cc globs.
    Truth.assertThat(issues).isEmpty();
  }

  @Test
  public void testBadJavaGlob() throws Exception {
    String errorMessage =
        findIssues("java_library(", "  name = \"foo\",", "  srcs = glob([\"**/*.java\"]),", ")")
            .toString();
    Truth.assertThat(errorMessage)
        .contains("go/build-style#globs Do not use recursive globs for Java source files.");
  }

  @Test
  public void testPredefinedJavaGlobVar() throws Exception {
    String errorMessage =
        findIssues(
                "my_var = \"**/*.java\"",
                "java_library(",
                "  name = \"foo\",",
                "  srcs = glob([my_var]),",
                ")")
            .toString();
    Truth.assertThat(errorMessage)
        .contains("go/build-style#globs Do not use recursive globs for Java source files.");
  }

  @Test
  public void testPostdefinedJavaGlobVar() throws Exception {
    String errorMessage =
        findIssues(
                "java_library(",
                "  name = \"foo\",",
                "  srcs = glob([my_var]),",
                ")",
                "my_var = \"**/*.java\"")
            .toString();
    Truth.assertThat(errorMessage)
        .contains("go/build-style#globs Do not use recursive globs for Java source files.");
  }
}
