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

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Implements SymlinkTreeAction by using the output service or by running an embedded script to
 * create the symlink tree.
 */
@ExecutionStrategy(contextType = SymlinkTreeActionContext.class)
public final class SymlinkTreeStrategy implements SymlinkTreeActionContext {
  private static final Logger logger = Logger.getLogger(SymlinkTreeStrategy.class.getName());

  private final OutputService outputService;
  private final BinTools binTools;

  public SymlinkTreeStrategy(OutputService outputService, BinTools binTools) {
    this.outputService = outputService;
    this.binTools = binTools;
  }

  @Override
  public void createSymlinks(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    actionExecutionContext.getEventHandler().post(new RunningActionEvent(action, "local"));
    try (AutoProfiler p =
        AutoProfiler.logged(
            "running " + action.prettyPrint(), logger, /*minTimeForLoggingInMilliseconds=*/ 100)) {
      try {
        if (outputService != null && outputService.canCreateSymlinkTree()) {
          Map<PathFragment, Path> symlinks = null;
          if (action.getRunfiles() != null) {
            try {
              symlinks =
                  Maps.transformValues(
                      action
                          .getRunfiles()
                          .getRunfilesInputs(
                              actionExecutionContext.getEventHandler(),
                              action.getOwner().getLocation(),
                              actionExecutionContext.getPathResolver()),
                      (artifact) -> artifact.getPath());
            } catch (IOException e) {
              throw new EnvironmentalExecException(e);
            }
          }
          outputService.createSymlinkTree(
              actionExecutionContext.getInputPath(action.getInputManifest()),
              symlinks,
              actionExecutionContext.getInputPath(action.getOutputManifest()),
              action.isFilesetTree(),
              action.getOutputManifest().getExecPath().getParentDirectory());
        } else if (!action.enableRunfiles()) {
          createSymlinkTreeHelper(action, actionExecutionContext).copyManifest();
        } else {
          Map<String, String> resolvedEnv = new LinkedHashMap<>();
          action.getEnvironment().resolve(resolvedEnv, actionExecutionContext.getClientEnv());
          createSymlinkTreeHelper(action, actionExecutionContext)
              .createSymlinksUsingCommand(
                  actionExecutionContext.getExecRoot(),
                  binTools,
                  resolvedEnv,
                  actionExecutionContext.getFileOutErr());
        }
      } catch (ExecException e) {
        throw e.toActionExecutionException(
            action.getProgressMessage(), actionExecutionContext.getVerboseFailures(), action);
      }
    }
  }

  private static SymlinkTreeHelper createSymlinkTreeHelper(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext) {
    return new SymlinkTreeHelper(
        actionExecutionContext.getInputPath(action.getInputManifest()),
        actionExecutionContext.getInputPath(action.getOutputManifest()).getParentDirectory(),
        action.isFilesetTree());
  }
}
