// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.testutil.LoggingTerminalWriter;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

/**
 * Tests {@link PositionAwareAnsiTerminalWriter}.
 */
@RunWith(JUnit4.class)
public class PositionAwareAnsiTerminalWriterTest {
  static final String NL = LoggingTerminalWriter.NEWLINE;
  static final String OK = LoggingTerminalWriter.OK;
  static final String FAIL = LoggingTerminalWriter.FAIL;
  static final String NORMAL = LoggingTerminalWriter.NORMAL;

  @Test
  public void positionSimple() throws IOException {
    final String sample = "lorem ipsum...";
    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.append(sample);

    assertEquals(sample.length(), terminalWriter.getPosition());
    assertEquals(sample, loggingTerminalWriter.getTranscript());
  }

  @Test
  public void positionTwoLines() throws IOException {
    final String firstLine = "lorem ipsum...";
    final String secondLine = "foo bar baz";

    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.append(firstLine);
    assertEquals(firstLine.length(), terminalWriter.getPosition());
    terminalWriter.newline();
    assertEquals(0, terminalWriter.getPosition());
    terminalWriter.append(secondLine);
    assertEquals(secondLine.length(), terminalWriter.getPosition());
    terminalWriter.newline();
    assertEquals(0, terminalWriter.getPosition());
    assertEquals(firstLine + NL + secondLine + NL, loggingTerminalWriter.getTranscript());
  }

  @Test
  public void positionNewlineTranslated() throws IOException {
    final String firstLine = "lorem ipsum...";
    final String secondLine = "foo bar baz";

    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.append(firstLine + "\n" + secondLine);
    assertEquals(secondLine.length(), terminalWriter.getPosition());
    terminalWriter.append("\n");
    assertEquals(0, terminalWriter.getPosition());
    assertEquals(firstLine + NL + secondLine + NL, loggingTerminalWriter.getTranscript());
  }

  @Test
  public void passThrough() throws IOException {
    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter
        .append("abc")
        .okStatus()
        .append("ok")
        .failStatus()
        .append("fail")
        .normal()
        .append("normal");
    assertEquals(
        "abc" + OK + "ok" + FAIL + "fail" + NORMAL + "normal",
        loggingTerminalWriter.getTranscript());
  }

  @Test
  public void highlightNospace() throws IOException {
    final String sample = "lorem ipsum...";

    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.failStatus();
    assertEquals(0, terminalWriter.getPosition());
    terminalWriter.append(sample);
    assertEquals(sample.length(), terminalWriter.getPosition());
    assertEquals(FAIL + sample, loggingTerminalWriter.getTranscript());
  }
}
