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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
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
public final class SymlinkTreeStrategy implements SymlinkTreeActionContext {
  private static final Logger logger = Logger.getLogger(SymlinkTreeStrategy.class.getName());

  @VisibleForTesting
  static final Function<Artifact, PathFragment> TO_PATH =
      (artifact) -> artifact == null ? null : artifact.getPath().asFragment();

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
          Path inputManifest =
              action.getInputManifest() == null
                  ? null
                  : actionExecutionContext.getInputPath(action.getInputManifest());
          Map<PathFragment, PathFragment> symlinks;
          if (action.getRunfiles() != null) {
            try {
              symlinks =
                  Maps.transformValues(runfilesToMap(action, actionExecutionContext), TO_PATH);
            } catch (IOException e) {
              throw new EnvironmentalExecException(e);
            }
          } else {
            Preconditions.checkState(action.isFilesetTree());
            Preconditions.checkNotNull(inputManifest);

            symlinks =
                SymlinkTreeHelper.processFilesetLinks(
                    actionExecutionContext
                        .getArtifactExpander()
                        .getFileset(action.getInputManifest()),
                    action.getFilesetRoot(),
                    actionExecutionContext.getExecRoot().asFragment());
          }

          outputService.createSymlinkTree(
              symlinks,
              action.getOutputManifest().getExecPath().getParentDirectory());

          createOutput(action, actionExecutionContext, inputManifest);
        } else if (!action.isRunfilesEnabled()) {
          createSymlinkTreeHelper(action, actionExecutionContext).copyManifest();
        } else if (action.getInputManifest() == null
            || (action.inprocessSymlinkCreation() && !action.isFilesetTree())) {
          try {
            Map<PathFragment, Artifact> runfiles = runfilesToMap(action, actionExecutionContext);
            createSymlinkTreeHelper(action, actionExecutionContext)
                .createSymlinksDirectly(
                    action.getOutputManifest().getPath().getParentDirectory(), runfiles);
          } catch (IOException e) {
            throw new EnvironmentalExecException(e).toActionExecutionException(action);
          }

          Path inputManifest =
              action.getInputManifest() == null
                  ? null
                  : actionExecutionContext.getInputPath(action.getInputManifest());
          createOutput(action, actionExecutionContext, inputManifest);
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

  private static Map<PathFragment, Artifact> runfilesToMap(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext) throws IOException {
    // This call outputs warnings about overlapping symlinks. However, this is already called by the
    // SourceManifestAction, so it can happen that we generate the warning twice. If the input
    // manifest is null, then we print the warning. Otherwise we assume that the
    // SourceManifestAction already printed it.
    return action
        .getRunfiles()
        .getRunfilesInputs(
            action.getInputManifest() == null ? actionExecutionContext.getEventHandler() : null,
            action.getOwner().getLocation(),
            actionExecutionContext.getPathResolver());
  }

  private static void createOutput(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext, Path inputManifest)
      throws EnvironmentalExecException {
    Path outputManifest = actionExecutionContext.getInputPath(action.getOutputManifest());
    if (inputManifest == null) {
      // If we don't have an input manifest, then create a file containing a fingerprint of
      // the runfiles object.
      Fingerprint fp = new Fingerprint();
      action.getRunfiles().fingerprint(fp);
      String hexDigest = fp.hexDigestAndReset();
      try {
        FileSystemUtils.writeContentAsLatin1(outputManifest, hexDigest);
      } catch (IOException e) {
        throw new EnvironmentalExecException(
            "Failed to link output manifest '" + outputManifest.getPathString() + "'", e);
      }
    } else {
      // Link output manifest on success. We avoid a file copy as these manifests may be
      // large. Note that this step has to come last because the OutputService may delete any
      // pre-existing symlink tree before creating a new one.
      try {
        outputManifest.createSymbolicLink(inputManifest);
      } catch (IOException e) {
        throw new EnvironmentalExecException(
            "Failed to link output manifest '" + outputManifest.getPathString() + "'", e);
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
