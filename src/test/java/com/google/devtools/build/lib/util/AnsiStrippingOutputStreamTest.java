// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * Tests for {@link AnsiStrippingOutputStream}.
 */
@RunWith(JUnit4.class)
public class AnsiStrippingOutputStreamTest {
  ByteArrayOutputStream output;
  PrintStream input;

  private static final String ESCAPE = "\u001b[";

  @Before
  public final void createStreams() throws Exception  {
    output = new ByteArrayOutputStream();
    OutputStream inputStream = new AnsiStrippingOutputStream(output);
    input = new PrintStream(inputStream);
  }

  private String getOutput(String... fragments) throws Exception {
    for (String fragment: fragments) {
      input.print(fragment);
    }

    return new String(output.toByteArray(), "ISO8859-1");
  }

  @Test
  public void doesNotFailHorribly() throws Exception {
    assertEquals("Love", getOutput("Love"));
  }

  @Test
  public void canStripAnsiCode() throws Exception {
    assertEquals("Love", getOutput(ESCAPE + "32mLove" + ESCAPE + "m"));
  }

  @Test
  public void recognizesAnsiCodeWhenBrokenUp() throws Exception {
    assertEquals("Love", getOutput("\u001b", "[", "mLove"));
  }

  @Test
  public void handlesOnlyEscCorrectly() throws Exception {
    assertEquals("\u001bLove", getOutput("\u001bLove"));
  }

  @Test
  public void handlesEscInPlaceOfControlCharCorrectly() throws Exception {
    assertEquals(ESCAPE + "31;42Love",
        getOutput(ESCAPE + "31;42" + ESCAPE + "1mLove"));
  }

  @Test
  public void handlesTwoEscapeSequencesCorrectly() throws Exception {
    assertEquals("Love",
        getOutput(ESCAPE + "32m" + ESCAPE + "1m" + "Love"));
  }

}
