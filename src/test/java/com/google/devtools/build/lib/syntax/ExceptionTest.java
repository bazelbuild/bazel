// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test class for exceptions.
 */
@RunWith(JUnit4.class)
public class ExceptionTest {

  @Test
  public void testEmptyMessage() throws Exception {
    Node dummyNode = Expression.parse(ParserInput.fromLines("DUMMY"));
    EvalExceptionWithStackTrace ex =
        new EvalExceptionWithStackTrace(new NullPointerException(), dummyNode);
    assertThat(ex)
        .hasMessageThat()
        .contains("Null Pointer: ExceptionTest.testEmptyMessage() in ExceptionTest.java:");
  }

  @Test
  public void testExceptionCause() throws Exception {
    IllegalArgumentException iae = new IllegalArgumentException("foo");
    EvalException ee = new EvalException(Location.BUILTIN, iae);

    runExceptionTest(iae, iae);
    runExceptionTest(ee, iae);
  }

  private static void runExceptionTest(Exception toThrow, Exception expectedCause)
      throws Exception {
    Node dummyNode = Expression.parse(ParserInput.fromLines("DUMMY"));
    EvalExceptionWithStackTrace ex = new EvalExceptionWithStackTrace(toThrow, dummyNode);
    assertThat(ex).hasCauseThat().isEqualTo(expectedCause);
  }
}
