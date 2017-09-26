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
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.skylark.skylint.Linter}. */
@RunWith(JUnit4.class)
public class LinterTest {
  @Test
  public void testAllCheckersAreRun() throws Exception {
    final String file =
        String.join(
            "\n",
            "_unusedVar = function()",
            "load(':foo.bzl', 'bar')",
            "def function():",
            "  return",
            "  'unreachable and unused string literal'");
    final String errorMessages =
        new Linter()
            .setFileContentsReader(p -> file.getBytes(StandardCharsets.ISO_8859_1))
            .lint(Paths.get("foo"))
            .toString();
    Truth.assertThat(errorMessages).contains("unreachable statement"); // control flow checker
    Truth.assertThat(errorMessages).contains("has no module docstring"); // docstring checker
    Truth.assertThat(errorMessages)
        .contains("load statement should be at the top"); // load statement checker
    Truth.assertThat(errorMessages)
        .contains("should be lower_snake_case"); // naming conventions checker
    Truth.assertThat(errorMessages)
        .contains("expression result not used"); // checker statements without effect
    Truth.assertThat(errorMessages).contains("unused definition of"); // usage checker
  }
}
