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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.MissingExpansionException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Implements SymlinkTreeAction by using the output service or by running an embedded script to
 * create the symlink tree.
 */
public final class SymlinkTreeStrategy implements SymlinkTreeActionContext {
  private static final Duration MIN_LOGGING = Duration.ofMillis(100);

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
        GoogleAutoProfilerUtils.logged("running " + action.prettyPrint(), MIN_LOGGING)) {
      try {
        if (outputService != null && outputService.canCreateSymlinkTree()) {
          Path inputManifest = actionExecutionContext.getInputPath(action.getInputManifest());
          Map<PathFragment, PathFragment> symlinks;
          if (action.getRunfiles() != null) {
            symlinks = Maps.transformValues(runfilesToMap(action), TO_PATH);
          } else {
            Preconditions.checkState(action.isFilesetTree());

            ImmutableList<FilesetOutputSymlink> filesetLinks;
            try {
              filesetLinks =
                  actionExecutionContext
                      .getArtifactExpander()
                      .getFileset(action.getInputManifest());
            } catch (MissingExpansionException e) {
              throw new IllegalStateException(e);
            }

            symlinks =
                SymlinkTreeHelper.processFilesetLinks(
                    filesetLinks,
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
            Map<PathFragment, Artifact> runfiles = runfilesToMap(action);
            createSymlinkTreeHelper(action, actionExecutionContext)
                .createSymlinksDirectly(
                    action.getOutputManifest().getPath().getParentDirectory(), runfiles);
          } catch (IOException e) {
            throw ActionExecutionException.fromExecException(
                new EnvironmentalExecException(e, Code.SYMLINK_TREE_CREATION_IO_EXCEPTION), action);
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
        throw ActionExecutionException.fromExecException(e, action);
      }
    }
  }

  private static Map<PathFragment, Artifact> runfilesToMap(SymlinkTreeAction action) {
    // This call outputs warnings about overlapping symlinks. However, this is already called by the
    // SourceManifestAction, so it can happen that we generate the warning twice. If the input
    // manifest is null, then we print the warning. Otherwise we assume that the
    // SourceManifestAction already printed it.
    return action
        .getRunfiles()
        .getRunfilesInputs(
            /* eventHandler= */ null,
            action.getOwner().getLocation(),
            action.getRepoMappingManifest());
  }

  private static void createOutput(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext, Path inputManifest)
      throws EnvironmentalExecException {
    Path outputManifest = actionExecutionContext.getInputPath(action.getOutputManifest());
    // Link output manifest on success. We avoid a file copy as these manifests may be
    // large. Note that this step has to come last because the OutputService may delete any
    // pre-existing symlink tree before creating a new one.
    try {
      outputManifest.createSymbolicLink(inputManifest);
    } catch (IOException e) {
      throw createLinkFailureException(outputManifest, e);
    }
  }

  private static SymlinkTreeHelper createSymlinkTreeHelper(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext) {
    return new SymlinkTreeHelper(
        actionExecutionContext.getInputPath(action.getInputManifest()),
        actionExecutionContext.getInputPath(action.getOutputManifest()).getParentDirectory(),
        action.isFilesetTree());
  }

  private static EnvironmentalExecException createLinkFailureException(
      Path outputManifest, IOException e) {
    return new EnvironmentalExecException(
        e,
        FailureDetail.newBuilder()
            .setMessage("Failed to link output manifest '" + outputManifest.getPathString() + "'")
            .setExecution(
                Execution.newBuilder().setCode(Code.SYMLINK_TREE_MANIFEST_LINK_IO_EXCEPTION))
            .build());
  }
}
