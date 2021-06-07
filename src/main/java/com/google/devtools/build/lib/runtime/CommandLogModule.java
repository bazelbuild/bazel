// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Supplier;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.io.OutputStream;
import java.util.logging.Level;

/** This module logs complete stdout / stderr output of Bazel to a local file. */
public class CommandLogModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private CommandEnvironment env;
  private OutputStream logOutputStream;

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
    builder.addInfoItems(new CommandLogInfoItem());
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.env = env;
  }

  @Override
  public OutErr getOutputListener() {
    Path commandLog = getCommandLogPath(env.getOutputBase());
    // Unlink old command log from previous build, if present.
    try {
      commandLog.delete();
    } catch (IOException ioException) {
      LoggingUtil.logToRemote(Level.WARNING, "Unable to delete command.log", ioException);
    }

    try {
      if (writeCommandLog(env.getRuntime()) && !"clean".equals(env.getCommandName())) {
        logOutputStream = commandLog.getOutputStream();
        return OutErr.create(logOutputStream, logOutputStream);
      }
    } catch (IOException ioException) {
      LoggingUtil.logToRemote(Level.WARNING, "Unable to delete or open command.log", ioException);
    }
    return null;
  }

  static boolean writeCommandLog(BlazeRuntime runtime) {
    OptionsParsingResult startupOptionsProvider = runtime.getStartupOptionsProvider();
    return startupOptionsProvider.getOptions(BlazeServerStartupOptions.class).writeCommandLog;
  }

  /**
   * For a given output_base directory, returns the command log file path.
   */
  static Path getCommandLogPath(Path outputBase) {
    return outputBase.getRelative("command.log");
  }

  @Override
  public void commandComplete() {
    CommandEnvironment localEnv = this.env;
    this.env = null;
    if (logOutputStream != null) {
      try {
        logOutputStream.flush();
        logOutputStream.close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("I/O exception closing log");
        String msg = "I/O exception closing log: " + e.getMessage();
        if (localEnv != null) {
          localEnv.getReporter().handle(Event.error(msg));
        } else {
          System.err.println(msg);
        }
      } finally {
        logOutputStream = null;
      }
    }
  }

  /**
   * Info item for the command log
   */
  public static final class CommandLogInfoItem extends InfoItem {
    public CommandLogInfoItem() {
      super(
          "command_log",
          "Location of the log containing the output from the build commands.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(getCommandLogPath(env.getRuntime().getWorkspace().getOutputBase()));
    }
  }
}
