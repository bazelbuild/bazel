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

import java.io.IOException;

/**
 * An {@link AnsiTerminalWriter} that just generates a transcript of the events it was exposed of.
 */
public class LoggingTerminalWriter implements AnsiTerminalWriter {
  // Strings for recording the non-append calls
  public static final String NEWLINE = "[NL]";
  public static final String OK = "[OK]";
  public static final String FAIL = "[FAIL]";
  public static final String NORMAL = "[NORMAL]";

  private String transcript;
  private final boolean discardHighlight;

  public LoggingTerminalWriter(boolean discardHighlight) {
    this.transcript = "";
    this.discardHighlight = discardHighlight;
  }

  public LoggingTerminalWriter() {
    this(false);
  }

  /** Clears the stored transcript; mostly useful for testing purposes. */
  public void reset() {
    transcript = "";
  }

  @Override
  public AnsiTerminalWriter append(String text) throws IOException {
    transcript += text;
    return this;
  }

  @Override
  public AnsiTerminalWriter newline() throws IOException {
    if (!discardHighlight) {
      transcript += NEWLINE;
    } else {
      transcript += "\n";
    }
    return this;
  }

  @Override
  public AnsiTerminalWriter okStatus() throws IOException {
    if (!discardHighlight) {
      transcript += OK;
    }
    return this;
  }

  @Override
  public AnsiTerminalWriter failStatus() throws IOException {
    if (!discardHighlight) {
      transcript += FAIL;
    }
    return this;
  }

  @Override
  public AnsiTerminalWriter normal() throws IOException {
    if (!discardHighlight) {
      transcript += NORMAL;
    }
    return this;
  }

  public String getTranscript() {
    return transcript;
  }
}
