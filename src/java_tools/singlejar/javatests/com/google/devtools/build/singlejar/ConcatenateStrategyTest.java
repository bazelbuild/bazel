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

package com.google.devtools.build.singlejar;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * Unit tests for {@link ConcatenateStrategy}.
 */
@RunWith(JUnit4.class)
public class ConcatenateStrategyTest {

  private String merge(String... inputs) throws IOException {
    return mergeInternal(true, inputs);
  }

  private String mergeNoNewLine(String... inputs) throws IOException {
    return mergeInternal(false, inputs);
  }

  private String mergeInternal(boolean appendNewLine, String... inputs) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ConcatenateStrategy strategy = new ConcatenateStrategy(appendNewLine);
    for (String input : inputs) {
      strategy.merge(new ByteArrayInputStream(input.getBytes(UTF_8)), out);
    }
    strategy.finish(out);
    return new String(out.toByteArray(), UTF_8);
  }

  @Test
  public void testSingleInput() throws IOException {
    assertEquals("a", merge("a"));
    assertEquals("a", mergeNoNewLine("a"));
  }

  @Test
  public void testTwoInputs() throws IOException {
    assertEquals("a\nb", merge("a\n", "b"));
    assertEquals("a\nb", mergeNoNewLine("a\n", "b"));
  }

  @Test
  public void testAutomaticNewline() throws IOException {
    assertEquals("a\nb", merge("a", "b"));
    assertEquals("ab", mergeNoNewLine("a", "b"));
  }

  @Test
  public void testAutomaticNewlineAndEmptyFile() throws IOException {
    assertEquals("a\nb", merge("a", "", "b"));
    assertEquals("ab", mergeNoNewLine("a", "", "b"));
  }
}
