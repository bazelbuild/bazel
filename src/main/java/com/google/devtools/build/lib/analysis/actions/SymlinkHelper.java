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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;

/** Bunch of static methods common to Symlink and SolibSymlink action */
public final class SymlinkHelper {

  public static final String POSIX_SYMLINK_COMMAND = "ln";

  /** Creates a symbolic link using the command line for posix systems. */
  protected static CommandLine getSymlinkCommandLine(
      String relativeLinkName, String absoluteTarget) {
    Preconditions.checkNotNull(relativeLinkName);
    Preconditions.checkNotNull(absoluteTarget);
    Preconditions.checkArgument(!relativeLinkName.isEmpty());
    Preconditions.checkArgument(!absoluteTarget.isEmpty());
    return CommandLine.of(
        ImmutableList.of(
            POSIX_SYMLINK_COMMAND,
            "-s", // make symbolic links
            absoluteTarget,
            relativeLinkName));
  }

  protected static Spawn createSpawn(
      AbstractAction action,
      ActionExecutionContext actionExecutionContext,
      String relativeLinkName,
      String absoluteTarget)
      throws ActionExecutionException {
    try {
      CommandLine symlinkCommandLine = getSymlinkCommandLine(relativeLinkName, absoluteTarget);
      return new SimpleSpawn(
          action,
          ImmutableList.copyOf(symlinkCommandLine.arguments()),
          actionExecutionContext.getClientEnv(),
          ImmutableMap.of(ExecutionRequirements.NO_REMOTE_EXEC, ""),
          action.getInputs(),
          action.getOutputs(),
          ResourceSet.ZERO);
    } catch (CommandLineExpansionException e) {
      throw new ActionExecutionException(
          "failed to generate symlink command for rule '"
              + action.getOwner().getLabel()
              + ": "
              + e.getMessage(),
          action,
          /* catastrophe= */ false);
    }
  }

  public static SpawnContinuation createFirst(
      AbstractAction action, ActionExecutionContext ctx, Artifact symlink, Path target)
      throws InterruptedException, ActionExecutionException {
    SpawnContinuation first;
    if (OS.getCurrent() == OS.WINDOWS) {
      Path symlinkPath = ctx.getInputPath(symlink);
      // For windows we use only LocalSymlinkStrategy and consequently no remote caching
      return ctx.getContext(SymlinkContext.class).createSymlink(symlinkPath, target);
    } else {
      Spawn spawn = createSpawn(action, ctx, symlink.getExecPathString(), target.toString());
      return ctx.getContext(SpawnStrategy.class).beginExecution(spawn, ctx);
    }
  }
}
