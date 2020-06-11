// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdebug.module;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.starlarkdebug.server.StarlarkDebugServer;
import com.google.devtools.build.lib.syntax.Debug;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/** Blaze module for setting up Starlark debugging. */
public final class StarlarkDebuggerModule extends BlazeModule {
  @Override
  public void beforeCommand(CommandEnvironment env) {
    // Conditionally enable debugging
    StarlarkDebuggerOptions buildOptions =
        env.getOptions().getOptions(StarlarkDebuggerOptions.class);
    boolean enabled = buildOptions != null && buildOptions.debugStarlark;
    if (enabled) {
      initializeDebugging(
          env.getReporter(), buildOptions.debugServerPort, buildOptions.verboseLogs);
    } else {
      disableDebugging();
    }
  }

  @Override
  public void afterCommand() {
    disableDebugging();
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(StarlarkDebuggerOptions.class)
        : ImmutableList.of();
  }

  @Override
  public void blazeShutdown() {
    disableDebugging();
  }

  @Override
  public void blazeShutdownOnCrash() {
    disableDebugging();
  }

  private static void initializeDebugging(Reporter reporter, int debugPort, boolean verboseLogs) {
    try {
      StarlarkDebugServer server =
          StarlarkDebugServer.createAndWaitForConnection(reporter, debugPort, verboseLogs);
      Debug.setDebugger(server);
    } catch (IOException e) {
      reporter.handle(Event.error("Error while setting up the debug server: " + e.getMessage()));
    }
  }

  private static void disableDebugging() {
    Debug.setDebugger(null);
  }
}
