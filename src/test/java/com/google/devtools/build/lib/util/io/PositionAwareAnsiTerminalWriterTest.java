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

import static com.google.common.truth.Truth.assertThat;

import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link PositionAwareAnsiTerminalWriter}. */
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

    assertThat(terminalWriter.getPosition()).isEqualTo(sample.length());
    assertThat(loggingTerminalWriter.getTranscript()).isEqualTo(sample);
  }

  @Test
  public void positionTwoLines() throws IOException {
    final String firstLine = "lorem ipsum...";
    final String secondLine = "foo bar baz";

    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.append(firstLine);
    assertThat(terminalWriter.getPosition()).isEqualTo(firstLine.length());
    terminalWriter.newline();
    assertThat(terminalWriter.getPosition()).isEqualTo(0);
    terminalWriter.append(secondLine);
    assertThat(terminalWriter.getPosition()).isEqualTo(secondLine.length());
    terminalWriter.newline();
    assertThat(terminalWriter.getPosition()).isEqualTo(0);
    assertThat(loggingTerminalWriter.getTranscript()).isEqualTo(firstLine + NL + secondLine + NL);
  }

  @Test
  public void positionNewlineTranslated() throws IOException {
    final String firstLine = "lorem ipsum...";
    final String secondLine = "foo bar baz";

    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.append(firstLine + "\n" + secondLine);
    assertThat(terminalWriter.getPosition()).isEqualTo(secondLine.length());
    terminalWriter.append("\n");
    assertThat(terminalWriter.getPosition()).isEqualTo(0);
    assertThat(loggingTerminalWriter.getTranscript()).isEqualTo(firstLine + NL + secondLine + NL);
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
    assertThat(loggingTerminalWriter.getTranscript())
        .isEqualTo("abc" + OK + "ok" + FAIL + "fail" + NORMAL + "normal");
  }

  @Test
  public void highlightNospace() throws IOException {
    final String sample = "lorem ipsum...";

    LoggingTerminalWriter loggingTerminalWriter = new LoggingTerminalWriter();
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(loggingTerminalWriter);

    terminalWriter.failStatus();
    assertThat(terminalWriter.getPosition()).isEqualTo(0);
    terminalWriter.append(sample);
    assertThat(terminalWriter.getPosition()).isEqualTo(sample.length());
    assertThat(loggingTerminalWriter.getTranscript()).isEqualTo(FAIL + sample);
  }
}
