// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * Implementation of {@link ImportantOutputHandler} for Build without the Bytes.
 *
 * <p>Any output that cannot be confirmed to still exist in remote cache results in rewinding.
 *
 * <p>The lifetime of an instance is a single build.
 */
public final class RemoteImportantOutputHandler implements ImportantOutputHandler {
  private final WalkableGraph graph;
  private final RemoteOutputChecker remoteOutputChecker;
  private final ActionInputPrefetcher actionInputPrefetcher;

  public RemoteImportantOutputHandler(
      WalkableGraph graph,
      RemoteOutputChecker remoteOutputChecker,
      ActionInputPrefetcher actionInputPrefetcher) {
    this.graph = graph;
    this.remoteOutputChecker = remoteOutputChecker;
    this.actionInputPrefetcher = actionInputPrefetcher;
  }

  @Override
  public LostArtifacts processOutputsAndGetLostArtifacts(
      Iterable<Artifact> importantOutputs,
      InputMetadataProvider importantMetadataProvider,
      InputMetadataProvider fullMetadataProvider)
      throws ImportantOutputException, InterruptedException {
    // Use the full metadata provider since we want to include runfiles trees.
    try {
      ensureToplevelArtifacts(importantOutputs, fullMetadataProvider);
    } catch (IOException e) {
      if (e instanceof BulkTransferException bulkTransferException) {
        var lostArtifacts = bulkTransferException.getLostArtifacts(fullMetadataProvider::getInput);
        if (!lostArtifacts.isEmpty()) {
          return lostArtifacts;
        }
      }
      throw new ImportantOutputException(
          e,
          FailureDetail.newBuilder()
              .setMessage(e.getMessage())
              .setRemoteExecution(
                  RemoteExecution.newBuilder()
                      .setCode(RemoteExecution.Code.TOPLEVEL_OUTPUTS_DOWNLOAD_FAILURE)
                      .build())
              .build());
    }
    return LostArtifacts.EMPTY;
  }

  @Override
  public LostArtifacts processRunfilesAndGetLostArtifacts(
      PathFragment runfilesDir,
      Map<PathFragment, Artifact> runfiles,
      InputMetadataProvider metadataProvider,
      String inputManifestExtension) {
    throw new UnsupportedOperationException(
        "Unused in Bazel, runfiles are processed in processOutputsAndGetLostArtifacts");
  }

  @Override
  public void processTestOutputs(Collection<Path> testOutputs) {
    // TODO: Either ensure that test outputs are never lost or implement a way to rewind them.
  }

  @Override
  public void processWorkspaceStatusOutputs(Path stableOutput, Path volatileOutput) {}

  private void ensureToplevelArtifacts(
      Iterable<Artifact> importantArtifacts, InputMetadataProvider metadataProvider)
      throws IOException, InterruptedException {
    var futures = new ArrayList<ListenableFuture<Void>>();

    for (var artifact : importantArtifacts) {
      downloadArtifact(metadataProvider, artifact, futures);
    }

    for (var runfileTree : metadataProvider.getRunfilesTrees()) {
      for (var artifact : runfileTree.getArtifacts().toList()) {
        downloadArtifact(metadataProvider, artifact, futures);
      }
    }

    // TODO: Only wait for failed futures to complete as long as they can all be explained by
    // lost outputs.
    try {
      var unused = Utils.mergeBulkTransfer(futures).get();
    } catch (ExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), IOException.class);
      Throwables.throwIfInstanceOf(e.getCause(), InterruptedException.class);
      Throwables.throwIfUnchecked(e.getCause());
      throw new IllegalStateException(e.getCause());
    }
  }

  private void downloadArtifact(
      InputMetadataProvider metadataProvider,
      Artifact artifact,
      List<ListenableFuture<Void>> futures)
      throws IOException, InterruptedException {
    if (!(artifact instanceof DerivedArtifact derivedArtifact)) {
      return;
    }

    // Metadata can be null during error bubbling, only download outputs that are already
    // generated. b/342188273
    if (artifact.isTreeArtifact()) {
      var treeArtifactValue = metadataProvider.getTreeMetadata(artifact);
      if (treeArtifactValue == null) {
        return;
      }

      var filesToDownload = new ArrayList<TreeFileArtifact>(treeArtifactValue.getChildren().size());
      for (var entry : treeArtifactValue.getChildValues().entrySet()) {
        if (remoteOutputChecker.shouldDownloadOutput(entry.getKey(), entry.getValue())) {
          filesToDownload.add(entry.getKey());
        }
      }
      if (!filesToDownload.isEmpty()) {
        futures.add(
            actionInputPrefetcher.prefetchFiles(
                // derivedArtifact's generating action may be an action template, which doesn't
                // implement the required ActionExecutionMetadata.
                getGeneratingAction(filesToDownload.getFirst()),
                filesToDownload,
                metadataProvider,
                ActionInputPrefetcher.Priority.LOW,
                ActionInputPrefetcher.Reason.OUTPUTS));
      }
    } else {
      FileArtifactValue metadata = metadataProvider.getInputMetadata(artifact);
      if (metadata == null) {
        return;
      }

      if (remoteOutputChecker.shouldDownloadOutput(artifact, metadata)) {
        futures.add(
            actionInputPrefetcher.prefetchFiles(
                getGeneratingAction(derivedArtifact),
                ImmutableList.of(artifact),
                metadataProvider,
                ActionInputPrefetcher.Priority.LOW,
                ActionInputPrefetcher.Reason.OUTPUTS));
      }
    }
  }

  private ActionExecutionMetadata getGeneratingAction(DerivedArtifact artifact)
      throws InterruptedException {
    var action = Actions.getGeneratingAction(graph, artifact);
    Preconditions.checkState(
        action instanceof ActionExecutionMetadata,
        "generating action for artifact %s is not an ActionExecutionMetadata, but %s",
        artifact,
        action != null ? action.getClass() : null);
    return (ActionExecutionMetadata) action;
  }
}
