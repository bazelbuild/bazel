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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.OutErr;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * An experimental new output stream.
 */
public class ExperimentalEventHandler extends BlazeCommandEventHandler {
  private static Logger LOG = Logger.getLogger(ExperimentalEventHandler.class.getName());

  private final AnsiTerminal terminal;

  public final int terminalWidth;

  public ExperimentalEventHandler(OutErr outErr, BlazeCommandEventHandler.Options options) {
    super(outErr, options);
    this.terminal = new AnsiTerminal(outErr.getErrorStream());
    this.terminalWidth = (options.terminalColumns > 0 ? options.terminalColumns : 80);
  }

  @Override
  public void handle(Event event) {}

  public void resetTerminal() {
    try {
      terminal.resetTerminal();
    } catch (IOException e) {
      LOG.warning("IO Error writing to user terminal: " + e);
    }
  }
}
