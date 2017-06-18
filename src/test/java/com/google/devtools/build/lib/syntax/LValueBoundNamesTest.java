// Copyright 2006 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.truth.Truth;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link LValue#boundNames()}. */
@RunWith(JUnit4.class)
public class LValueBoundNamesTest {

  @Test
  public void simpleAssignment() {
    assertBoundNames("x = 1", "x");
  }

  @Test
  public void listAssignment() {
    assertBoundNames("x, y = 1", "x", "y");
  }

  @Test
  public void complexListAssignment() {
    assertBoundNames("x, [y] = 1", "x", "y");
  }

  @Test
  public void arrayElementAssignment() {
    assertBoundNames("x[1] = 1");
  }

  @Test
  public void complexListAssignment2() {
    assertBoundNames("[[x], y], [z, w[1]] = 1", "x", "y", "z");
  }

  private static void assertBoundNames(String assignement, String... boundNames) {
    BuildFileAST buildFileAST = BuildFileAST
        .parseSkylarkString(Environment.FAIL_FAST_HANDLER, assignement);
    LValue lValue = ((AssignmentStatement) buildFileAST.getStatements().get(0)).getLValue();
    Truth.assertThat(lValue.boundNames()).containsExactlyElementsIn(Arrays.asList(boundNames));
  }
}
