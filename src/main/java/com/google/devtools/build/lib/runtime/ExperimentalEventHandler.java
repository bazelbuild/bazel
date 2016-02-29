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
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LineWrappingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.OutErr;

import java.io.IOException;
import java.io.OutputStream;
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
  public synchronized void handle(Event event) {
    try {
      if (debugAllEvents) {
        // Debugging only: show all events visible to the new UI.
        clearProgressBar();
        terminal.flush();
        outErr.getOutputStream().write((event + "\n").getBytes(StandardCharsets.UTF_8));
        outErr.getOutputStream().flush();
        addProgressBar();
        terminal.flush();
      } else {
        switch (event.getKind()) {
          case STDOUT:
          case STDERR:
            clearProgressBar();
            terminal.flush();
            OutputStream stream =
                event.getKind() == EventKind.STDOUT
                    ? outErr.getOutputStream()
                    : outErr.getErrorStream();
            stream.write(event.getMessageBytes());
            stream.write(new byte[] {10, 13});
            stream.flush();
            addProgressBar();
            terminal.flush();
            break;
          case ERROR:
          case WARNING:
          case INFO:
            clearProgressBar();
            setEventKindColor(event.getKind());
            terminal.writeString(event.getKind() + ": ");
            terminal.resetTerminal();
            if (event.getLocation() != null) {
              terminal.writeString(event.getLocation() + ": ");
            }
            if (event.getMessage() != null) {
              terminal.writeString(event.getMessage());
            }
            crlf();
            addProgressBar();
            terminal.flush();
            break;
        }
      }
    } catch (IOException e) {
      LOG.warning("IO Error writing to output stream: " + e);
    }
  }

  private void setEventKindColor(EventKind kind) throws IOException {
    switch (kind) {
      case ERROR:
        terminal.textRed();
        terminal.textBold();
        break;
      case WARNING:
        terminal.textMagenta();
        break;
      case INFO:
        terminal.textGreen();
        break;
    }
  }

  @Subscribe
  public void buildStarted(BuildStartingEvent event) {
    stateTracker.buildStarted(event);
    refresh();
  }

  @Subscribe
  public void loadingComplete(LoadingPhaseCompleteEvent event) {
    stateTracker.loadingComplete(event);
    refresh();
  }

  @Subscribe
  public void analysisComplete(AnalysisPhaseCompleteEvent event) {
    stateTracker.analysisComplete(event);
    refresh();
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    stateTracker.buildComplete(event);
    refresh();
  }

  @Subscribe
  public void actionStarted(ActionStartedEvent event) {
    stateTracker.actionStarted(event);
    refresh();
  }

  @Subscribe
  public void actionCompletion(ActionCompletionEvent event) {
    stateTracker.actionCompletion(event);
    refresh();
  }

  private synchronized void refresh() {
    try {
      clearProgressBar();
      addProgressBar();
      terminal.flush();
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

  private void crlf() throws IOException {
    terminal.cr();
    terminal.writeString("\n");
  }

  private void addProgressBar() throws IOException {
    LineCountingAnsiTerminalWriter terminalWriter = new LineCountingAnsiTerminalWriter(terminal);
    stateTracker.writeProgressBar(
        new LineWrappingAnsiTerminalWriter(terminalWriter, terminalWidth - 1));
    terminalWriter.newline();
    numLinesProgressBar = terminalWriter.getWrittenLines();
  }

  private class LineCountingAnsiTerminalWriter implements AnsiTerminalWriter {

    private final AnsiTerminal terminal;
    private int lineCount;

    LineCountingAnsiTerminalWriter(AnsiTerminal terminal) {
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
}
