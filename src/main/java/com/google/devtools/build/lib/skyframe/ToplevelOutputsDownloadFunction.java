// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.ImportantOutputException;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.LostArtifacts;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.analysis.ConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingArtifactValue;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Downloads those top-level outputs of a single top-level target (or aspect) that the current
 * invocation's download policy wants available locally but that only exist as remote metadata,
 * e.g. because their generating action had an action cache hit.
 *
 * <p>The returned {@link ToplevelOutputsDownloadValue} records the outputs whose local presence is
 * not tracked by Skyframe otherwise, so that {@link FilesystemValueChecker} can invalidate this
 * node when the local state diverges (e.g. because the user deleted a downloaded output) and its
 * reevaluation can restore the file.
 */
final class ToplevelOutputsDownloadFunction implements SkyFunction {
  private final SkyframeActionExecutor skyframeActionExecutor;

  ToplevelOutputsDownloadFunction(SkyframeActionExecutor skyframeActionExecutor) {
    this.skyframeActionExecutor = skyframeActionExecutor;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    var key = (ToplevelOutputsDownloadValue.Key) skyKey;
    var importantOutputHandler =
        skyframeActionExecutor.getActionContextRegistry().getContext(ImportantOutputHandler.class);
    if (importantOutputHandler == null) {
      // Without an important output handler (e.g. without Skymeld or in a non-remote build that
      // still has a download policy), there is nothing to download.
      return new ToplevelOutputsDownloadValue(ImmutableMap.of());
    }

    Pair<ConfiguredObjectValue, ArtifactsToBuild> valueAndArtifactsToBuild =
        CompletionFunction.getValueAndArtifactsToBuild(key, env);
    if (valueAndArtifactsToBuild == null) {
      return null;
    }
    ArtifactsToBuild artifactsToBuild = valueAndArtifactsToBuild.second;

    ImmutableList<Artifact> allArtifacts = artifactsToBuild.getAllArtifacts().toList();
    SkyframeLookupResult inputDeps = env.getValuesAndExceptions(Artifact.keys(allArtifacts));

    ActionInputMap inputMap = new ActionInputMap(allArtifacts.size());
    for (Artifact input : allArtifacts) {
      try {
        SkyValue artifactValue =
            inputDeps.getOrThrow(
                Artifact.key(input), ActionExecutionException.class, SourceArtifactException.class);
        if (artifactValue == null || artifactValue instanceof MissingArtifactValue) {
          // Missing values are handled after the loop; error reporting is the responsibility of
          // the completion function depending on this node, which requests the same artifacts.
          continue;
        }
        ActionInputMapHelper.addToMap(
            inputMap, input, artifactValue, MetadataConsumerForMetrics.NO_OP);
      } catch (ActionExecutionException | SourceArtifactException e) {
        // Failed artifacts can't be downloaded. The completion function reports the failure.
      }
    }
    if (env.valuesMissing()) {
      return null;
    }

    ImmutableCollection<Artifact> importantArtifacts =
        artifactsToBuild.areAllOutputGroupsImportant()
            ? allArtifacts
            : artifactsToBuild.getImportantArtifacts().toSet();
    InputMetadataProvider metadataProvider = new ActionInputMetadataProvider(inputMap);
    try (var ignored =
        GoogleAutoProfilerUtils.profiledAndLogged(
            "Downloading top-level outputs for " + key.actionLookupKey().getLabel(),
            ProfilerTask.INFO,
            ImportantOutputHandler.LOG_THRESHOLD)) {
      LostArtifacts unusedLostOutputs =
          importantOutputHandler.processOutputsAndGetLostArtifacts(
              key.topLevelArtifactContext().expandFilesets()
                  ? importantArtifacts
                  : Iterables.filter(importantArtifacts, artifact -> !artifact.isFileset()),
              metadataProvider);
      // Lost outputs and download failures are currently detected and handled (via action
      // rewinding or build failure) by the completion function's own call into the important
      // output handler, which runs after this node has been evaluated.
    } catch (ImportantOutputException e) {
      // Handled by the completion function's own call into the important output handler.
    }

    return new ToplevelOutputsDownloadValue(
        collectMaterializedOutputs(importantArtifacts, metadataProvider));
  }

  /**
   * Collects the output files that are present in the local filesystem while the metadata tracked
   * for them in Skyframe is remote.
   */
  private static ImmutableMap<Artifact, FileContentsProxy> collectMaterializedOutputs(
      ImmutableCollection<Artifact> importantArtifacts, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    Map<Artifact, FileContentsProxy> materializedOutputs = new LinkedHashMap<>();

    for (Artifact artifact : importantArtifacts) {
      collectMaterializedOutputs(metadataProvider, artifact, materializedOutputs);
    }
    for (var runfilesTree : metadataProvider.getRunfilesTrees()) {
      for (var artifact : runfilesTree.getArtifacts().toList()) {
        collectMaterializedOutputs(metadataProvider, artifact, materializedOutputs);
      }
    }

    return ImmutableMap.copyOf(materializedOutputs);
  }

  private static void collectMaterializedOutputs(
      InputMetadataProvider metadataProvider,
      Artifact artifact,
      Map<Artifact, FileContentsProxy> materializedOutputs)
      throws InterruptedException {
    if (artifact.isFileset() || artifact.isRunfilesTree()) {
      return;
    }
    try {
      if (artifact.isTreeArtifact()) {
        var treeArtifactValue = metadataProvider.getTreeMetadata(artifact);
        if (treeArtifactValue == null) {
          return;
        }
        for (var entry : treeArtifactValue.getChildValues().entrySet()) {
          addIfMaterialized(entry.getKey(), entry.getValue(), materializedOutputs);
        }
      } else {
        FileArtifactValue metadata = metadataProvider.getInputMetadata(artifact);
        if (metadata == null) {
          return;
        }
        addIfMaterialized(artifact, metadata, materializedOutputs);
      }
    } catch (IOException e) {
      // An output whose metadata can't be read isn't recorded; a deletion of its local copy then
      // doesn't invalidate this node, which errs on the side of not redoing work.
    }
  }

  private static void addIfMaterialized(
      Artifact artifact,
      FileArtifactValue metadata,
      Map<Artifact, FileContentsProxy> materializedOutputs) {
    if (!metadata.isRemote() || metadata.isInMemoryOutput()) {
      return;
    }
    // A contents proxy on remote metadata is recorded when the file is materialized in the local
    // filesystem. It may be stale if the file has been deleted since a previous invocation
    // materialized it and the current invocation's download policy doesn't want it locally, so
    // only record files that are actually present.
    FileContentsProxy contentsProxy = metadata.getContentsProxy();
    if (contentsProxy == null || !artifact.getPath().exists()) {
      return;
    }
    materializedOutputs.put(artifact, contentsProxy);
  }
}
