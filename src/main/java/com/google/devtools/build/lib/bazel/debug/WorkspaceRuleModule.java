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
package com.google.devtools.build.lib.bazel.debug;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Workspaces;
import com.google.devtools.build.lib.server.FailureDetails.Workspaces.Code;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/** A module for logging workspace rule events */
public final class WorkspaceRuleModule extends BlazeModule {

  private AsynchronousFileOutputStream outFileStream;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    Reporter reporter = env.getReporter();
    EventBus eventBus = env.getEventBus();

    if (env.getOptions() == null || env.getOptions().getOptions(DebuggingOptions.class) == null) {
      reporter.handle(Event.error("Installation is corrupt: could not retrieve debugging options"));
      return;
    }

    PathFragment logFile =
        env.getOptions().getOptions(DebuggingOptions.class).workspaceRulesLogFile;
    if (logFile != null) {
      try {
        outFileStream =
            new AsynchronousFileOutputStream(env.getWorkingDirectory().getRelative(logFile));
      } catch (IOException e) {
        env.getReporter().handle(Event.error(e.getMessage()));
        env.getBlazeModuleEnvironment()
            .exit(
                new AbruptExitException(
                    createDetailedExitCode(
                        "Error initializing workspace rule log file.",
                        Code.WORKSPACES_LOG_INITIALIZATION_FAILURE)));
      }
      eventBus.register(this);
    }
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    if (outFileStream != null) {
      try {
        // Any AsynchronousFileOutputStream write failures get rethrown here.
        outFileStream.close();
      } catch (IOException e) {
        String message =
            e.getMessage() == null ? "Error writing workspace rule log file." : e.getMessage();
        throw new AbruptExitException(
            createDetailedExitCode(message, Code.WORKSPACES_LOG_WRITE_FAILURE), e);
      } finally {
        outFileStream = null;
      }
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(DebuggingOptions.class);
  }

  @Subscribe
  public void workspaceRuleEventReceived(WorkspaceRuleEvent event) {
    if (outFileStream != null) {
      outFileStream.write(event.getLogEvent());
    }
  }

  private static DetailedExitCode createDetailedExitCode(String message, Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setWorkspaces(Workspaces.newBuilder().setCode(detailedCode))
            .build());
  }
}
