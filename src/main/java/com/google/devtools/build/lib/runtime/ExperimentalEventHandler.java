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
package com.google.devtools.build.lib.runtime;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.OutErr;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

/**
 * An experimental new output stream.
 */
public class ExperimentalEventHandler extends BlazeCommandEventHandler {
  private static Logger LOG = Logger.getLogger(ExperimentalEventHandler.class.getName());

  private final AnsiTerminal terminal;
  private final boolean debugAllEvents;
  private final ExperimentalStateTracker stateTracker;
  private int numLinesProgressBar;

  public final int terminalWidth;

  public ExperimentalEventHandler(OutErr outErr, BlazeCommandEventHandler.Options options) {
    super(outErr, options);
    this.terminal = new AnsiTerminal(outErr.getErrorStream());
    this.terminalWidth = (options.terminalColumns > 0 ? options.terminalColumns : 80);
    this.debugAllEvents = options.experimentalUiDebugAllEvents;
    this.stateTracker = new ExperimentalStateTracker();
    this.numLinesProgressBar = 0;
  }

  @Override
  public void handle(Event event) {
    try {
      clearProgressBar();
      if (debugAllEvents) {
        // Debugging only: show all events visible to the new UI.
        terminal.flush();
        outErr.getOutputStream().write((event + "\n").getBytes(StandardCharsets.UTF_8));
        outErr.getOutputStream().flush();
      }
      // The actual new UI.
      addProgressBar();
      terminal.flush();
    } catch (IOException e) {
      LOG.warning("IO Error writing to output stream: " + e);
    }
  }

  @Subscribe
  public void loadingComplete(LoadingPhaseCompleteEvent event) {
    stateTracker.loadingComplete(event);
    try {
      clearProgressBar();
      terminal.textGreen();
      terminal.writeString("INFO:");
      terminal.resetTerminal();
      terminal.writeString(" " + event.getTargets().size() + " Target(s)\n");
      addProgressBar();
    } catch (IOException e) {
      LOG.warning("IO Error writing to output stream: " + e);
    }
  }

  public void resetTerminal() {
    try {
      terminal.resetTerminal();
    } catch (IOException e) {
      LOG.warning("IO Error writing to user terminal: " + e);
    }
  }

  private void clearProgressBar() throws IOException {
    for (int i = 0; i < numLinesProgressBar; i++) {
      terminal.cr();
      terminal.cursorUp(1);
      terminal.clearLine();
    }
    numLinesProgressBar = 0;
  }

  private void addProgressBar() throws IOException {
    LineCountingAnsiTerminalWriter terminalWriter = new LineCountingAnsiTerminalWriter(terminal);
    stateTracker.writeProgressBar(terminalWriter);
    terminalWriter.newline();
    numLinesProgressBar = terminalWriter.getWrittenLines();
  }

  private class LineCountingAnsiTerminalWriter implements AnsiTerminalWriter {

    private final AnsiTerminal terminal;
    private int lineCount;
    private int currentLineLength;

    LineCountingAnsiTerminalWriter(AnsiTerminal terminal) {
      this.terminal = terminal;
      this.lineCount = 0;
    }

    @Override
    public AnsiTerminalWriter append(String text) throws IOException {
      terminal.writeString(text);
      currentLineLength += text.length();
      return this;
    }

    @Override
    public AnsiTerminalWriter newline() throws IOException {
      terminal.cr();
      terminal.writeString("\n");
      // Besides the line ended by the newline() command, a line-shift can also happen
      // by a string longer than the terminal width being wrapped around; account for
      // this as well.
      lineCount += 1 + currentLineLength / terminalWidth;
      currentLineLength = 0;
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
}
