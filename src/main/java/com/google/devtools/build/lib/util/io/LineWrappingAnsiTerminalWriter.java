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
 * Wrap an {@link AnsiTerminalWriter} into one that breaks lines to use
 * at most a first given number of columns of the terminal. In this way,
 * all line breaks are predictable, even if we only have a lower bound
 * on the number of columns of the underlying terminal. To simplify copy
 * and paste of the terminal output, a continuation character is written
 * into the last usable column when we break a line. Additionally, newline
 * characters are translated into calls to the {@link AnsiTerminalWriter#newline()}
 * method.
 */
public class LineWrappingAnsiTerminalWriter implements AnsiTerminalWriter {

  private final AnsiTerminalWriter terminalWriter;
  private final int width;
  private final char continuationCharacter;
  private int position;

  public LineWrappingAnsiTerminalWriter(
      AnsiTerminalWriter terminalWriter, int width, char continuationCharacter) {
    this.terminalWriter = terminalWriter;
    this.width = width;
    this.continuationCharacter = continuationCharacter;
    this.position = 0;
  }

  public LineWrappingAnsiTerminalWriter(AnsiTerminalWriter terminalWriter, int width) {
    this(terminalWriter, width, '\\');
  }

  private void appendChar(char c) throws IOException {
    if (c == '\n') {
      terminalWriter.newline();
      position = 0;
    } else if (position + 2 < width) {
      terminalWriter.append(Character.toString(c));
      position++;
    } else {
      terminalWriter.append(new String(new char[] {c, continuationCharacter}));
      terminalWriter.newline();
      position = 0;
    }
  }

  @Override
  public AnsiTerminalWriter append(String text) throws IOException {
    for (int i = 0; i < text.length(); i++) {
      appendChar(text.charAt(i));
    }
    return this;
  }

  @Override
  public AnsiTerminalWriter newline() throws IOException {
    terminalWriter.newline();
    position = 0;
    return this;
  }

  @Override
  public AnsiTerminalWriter okStatus() throws IOException {
    terminalWriter.okStatus();
    return this;
  }

  @Override
  public AnsiTerminalWriter failStatus() throws IOException {
    terminalWriter.failStatus();
    return this;
  }

  @Override
  public AnsiTerminalWriter normal() throws IOException {
    terminalWriter.normal();
    return this;
  }
}
