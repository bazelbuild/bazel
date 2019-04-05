// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction.DeterministicWriter;
import com.google.devtools.build.lib.analysis.actions.FileWriteActionContext;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.logging.Logger;

/**
 * A strategy for executing an {@link AbstractFileWriteAction}.
 */
@ExecutionStrategy(name = { "local" }, contextType = FileWriteActionContext.class)
public final class FileWriteStrategy implements FileWriteActionContext {
  private static final Logger logger = Logger.getLogger(FileWriteStrategy.class.getName());
  public static final Class<FileWriteStrategy> TYPE = FileWriteStrategy.class;

  public FileWriteStrategy() {
  }

  @Override
  public SpawnContinuation beginWriteOutputToFile(
      AbstractAction action,
      ActionExecutionContext actionExecutionContext,
      DeterministicWriter deterministicWriter,
      boolean makeExecutable,
      boolean isRemotable)
      throws ExecException {
    actionExecutionContext.getEventHandler().post(new RunningActionEvent(action, "local"));
    // TODO(ulfjack): Consider acquiring local resources here before trying to write the file.
    try (AutoProfiler p =
        AutoProfiler.logged(
            "running write for action " + action.prettyPrint(),
            logger,
            /*minTimeForLoggingInMilliseconds=*/ 100)) {
      Path outputPath =
          actionExecutionContext.getInputPath(Iterables.getOnlyElement(action.getOutputs()));
      try {
        try (OutputStream out = new BufferedOutputStream(outputPath.getOutputStream())) {
          deterministicWriter.writeOutputFile(out);
        }
        if (makeExecutable) {
          outputPath.setExecutable(true);
        }
      } catch (IOException e) {
        throw new EnvironmentalExecException("IOException during file write", e);
      }
    }
    return SpawnContinuation.immediate();
  }
}
