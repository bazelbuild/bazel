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
 * Class providing the AnsiTerminalWriter interface from a terminal while additionally counting the
 * number of written lines.
 */
public class LineCountingAnsiTerminalWriter implements AnsiTerminalWriter {

  private final AnsiTerminal terminal;
  private int lineCount;

  public LineCountingAnsiTerminalWriter(AnsiTerminal terminal) {
    this.terminal = terminal;
    this.lineCount = 0;
  }

  @Override
  public AnsiTerminalWriter append(String text) throws IOException {
    terminal.writeString(text);
    return this;
  }

  @Override
  public AnsiTerminalWriter newline() throws IOException {
    terminal.cr();
    terminal.writeString("\n");
    lineCount++;
    return this;
  }

  @Override
  public AnsiTerminalWriter okStatus() throws IOException {
    terminal.textGreen();
    return this;
  }

  @Override
  public AnsiTerminalWriter failStatus() throws IOException {
    terminal.textRed();
    terminal.textBold();
    return this;
  }

  @Override
  public AnsiTerminalWriter normal() throws IOException {
    terminal.resetTerminal();
    return this;
  }

  public int getWrittenLines() throws IOException {
    return lineCount;
  }
}
