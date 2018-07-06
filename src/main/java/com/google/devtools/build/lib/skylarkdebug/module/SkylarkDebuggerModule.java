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

package com.google.devtools.build.lib.skylarkdebug.module;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skylarkdebug.server.SkylarkDebugServer;
import com.google.devtools.build.lib.syntax.DebugServerUtils;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/** Blaze module for setting up Skylark debugging. */
public final class SkylarkDebuggerModule extends BlazeModule {
  @Override
  public void beforeCommand(CommandEnvironment env) {
    // Conditionally enable debugging
    SkylarkDebuggerOptions buildOptions = env.getOptions().getOptions(SkylarkDebuggerOptions.class);
    boolean enabled = buildOptions != null && buildOptions.debugSkylark;
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
        ? ImmutableList.of(SkylarkDebuggerOptions.class)
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
      SkylarkDebugServer server =
          SkylarkDebugServer.createAndWaitForConnection(reporter, debugPort, verboseLogs);
      DebugServerUtils.initializeDebugServer(server);
    } catch (IOException e) {
      reporter.handle(Event.error("Error while setting up the debug server: " + e.getMessage()));
    }
  }

  private static void disableDebugging() {
    DebugServerUtils.disableDebugging();
  }
}
