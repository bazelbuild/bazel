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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionInputMapSink;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import javax.annotation.Nullable;

/** Static utilities for working with action inputs. */
final class ActionInputMapHelper {

  private ActionInputMapHelper() {}

  static void addToMap(
      ActionInputMapSink inputMap,
      Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
      Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsInsideRunfiles,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets,
      Artifact key,
      SkyValue value,
      Environment env,
      boolean requiresTreeMetadataWhenTreeFileIsInput)
      throws InterruptedException {
    addToMap(
        inputMap,
        expandedArtifacts,
        archivedTreeArtifacts,
        filesetsInsideRunfiles,
        topLevelFilesets,
        key,
        value,
        env,
        MetadataConsumerForMetrics.NO_OP,
        requiresTreeMetadataWhenTreeFileIsInput);
  }

  /**
   * Adds a value obtained by an Artifact skyvalue lookup to the action input map. May do Skyframe
   * lookups.
   */
  static void addToMap(
      ActionInputMapSink inputMap,
      Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
      Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsInsideRunfiles,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets,
      Artifact key,
      SkyValue value,
      Environment env,
      MetadataConsumerForMetrics consumer,
      boolean requiresTreeMetadataWhenTreeFileIsInput)
      throws InterruptedException {
    if (value instanceof RunfilesArtifactValue) {
      // Note: we don't expand the .runfiles/MANIFEST file into the inputs. The reason for that
      // being that the MANIFEST file contains absolute paths that don't work with remote execution.
      // Instead, the way the SpawnInputExpander expands runfiles is via the Runfiles class
      // which contains all artifacts in the runfiles tree minus the MANIFEST file.
      RunfilesArtifactValue runfilesArtifactValue = (RunfilesArtifactValue) value;
      runfilesArtifactValue.forEachFile(
          (artifact, metadata) -> {
            inputMap.put(artifact, metadata, /* depOwner= */ key);
            if (artifact.isFileset()) {
              ImmutableList<FilesetOutputSymlink> expandedFileset =
                  getFilesets(env, (SpecialArtifact) artifact);
              if (expandedFileset != null) {
                filesetsInsideRunfiles.put(artifact, expandedFileset);
                consumer.accumulate(expandedFileset);
              }
            } else {
              consumer.accumulate(metadata);
            }
          });
      runfilesArtifactValue.forEachTree(
          (treeArtifact, metadata) -> {
            expandTreeArtifactAndPopulateArtifactData(
                treeArtifact,
                metadata,
                expandedArtifacts,
                archivedTreeArtifacts,
                inputMap,
                /* depOwner= */ key);
            consumer.accumulate(metadata);
          });
      // We have to cache the "digest" of the aggregating value itself, because the action cache
      // checker may want it.
      inputMap.put(key, runfilesArtifactValue.getMetadata(), /* depOwner= */ key);
    } else if (value instanceof TreeArtifactValue) {
      TreeArtifactValue treeArtifactValue = (TreeArtifactValue) value;
      expandTreeArtifactAndPopulateArtifactData(
          key,
          treeArtifactValue,
          expandedArtifacts,
          archivedTreeArtifacts,
          inputMap,
          /*depOwner=*/ key);
      consumer.accumulate(treeArtifactValue);
    } else if (value instanceof ActionExecutionValue) {
      if (requiresTreeMetadataWhenTreeFileIsInput && key.isChildOfDeclaredDirectory()) {
        // Actions resulting from the expansion of an ActionTemplate consume only one of the files
        // in a tree artifact. However, the input prefetcher requires access to the tree metadata
        // to determine the prefetch location of a tree artifact materialized as a symlink
        // (cf. TreeArtifactValue#getMaterializationExecPath()).
        SpecialArtifact treeArtifact = key.getParent();
        TreeArtifactValue treeArtifactValue =
            ((ActionExecutionValue) value).getTreeArtifactValue(treeArtifact);
        inputMap.putTreeArtifact(treeArtifact, treeArtifactValue, /* depOwner= */ treeArtifact);
        consumer.accumulate(treeArtifactValue);
      }
      FileArtifactValue metadata = ((ActionExecutionValue) value).getExistingFileArtifactValue(key);
      inputMap.put(key, metadata, key);
      if (key.isFileset()) {
        ImmutableList<FilesetOutputSymlink> filesets = getFilesets(env, (SpecialArtifact) key);
        if (filesets != null) {
          topLevelFilesets.put(key, filesets);
          consumer.accumulate(filesets);
        }
      } else {
        consumer.accumulate(metadata);
      }
    } else {
      checkArgument(value instanceof FileArtifactValue, "Unexpected value %s", value);
      FileArtifactValue metadata = (FileArtifactValue) value;
      inputMap.put(key, metadata, /*depOwner=*/ key);
      consumer.accumulate(metadata);
    }
  }

  @Nullable
  private static ImmutableList<FilesetOutputSymlink> getFilesets(
      Environment env, SpecialArtifact actionInput) throws InterruptedException {
    checkState(actionInput.isFileset(), actionInput);
    ActionLookupData generatingActionKey = actionInput.getGeneratingActionKey();
    ActionLookupKey filesetActionLookupKey = generatingActionKey.getActionLookupKey();

    ActionLookupValue filesetActionLookupValue =
        (ActionLookupValue) env.getValue(filesetActionLookupKey);

    ActionAnalysisMetadata generatingAction =
        filesetActionLookupValue.getAction(generatingActionKey.getActionIndex());
    ActionLookupData filesetActionKey;

    if (generatingAction instanceof SymlinkAction) {
      DerivedArtifact outputManifest =
          (DerivedArtifact) generatingAction.getInputs().getSingleton();
      ActionLookupData manifestGeneratingKey = outputManifest.getGeneratingActionKey();
      checkState(
          manifestGeneratingKey.getActionLookupKey().equals(filesetActionLookupKey),
          "Mismatched actions and artifacts: %s %s %s %s",
          actionInput,
          outputManifest,
          filesetActionLookupKey,
          manifestGeneratingKey);
      ActionAnalysisMetadata symlinkTreeAction =
          filesetActionLookupValue.getAction(manifestGeneratingKey.getActionIndex());
      DerivedArtifact inputManifest =
          (DerivedArtifact) symlinkTreeAction.getInputs().getSingleton();
      ActionLookupData inputManifestGeneratingKey = inputManifest.getGeneratingActionKey();
      checkState(
          inputManifestGeneratingKey.getActionLookupKey().equals(filesetActionLookupKey),
          "Mismatched actions and artifacts: %s %s %s %s",
          actionInput,
          inputManifest,
          filesetActionLookupKey,
          inputManifestGeneratingKey);
      filesetActionKey = inputManifestGeneratingKey;
    } else {
      filesetActionKey = generatingActionKey;
    }

    // TODO(janakr: properly handle exceptions coming from here, or prove they can never happen in
    //  practice.
    ActionExecutionValue filesetValue = (ActionExecutionValue) env.getValue(filesetActionKey);
    if (filesetValue == null) {
      // At this point skyframe does not guarantee that the filesetValue will be ready, since
      // the current action does not directly depend on the outputs of the
      // SkyframeFilesetManifestAction whose ActionExecutionValue (filesetValue) is needed here.
      return null;
    }
    return filesetValue.getOutputSymlinks();
  }

  private static void expandTreeArtifactAndPopulateArtifactData(
      Artifact treeArtifact,
      TreeArtifactValue value,
      Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
      Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
      ActionInputMapSink inputMap,
      Artifact depOwner) {
    if (TreeArtifactValue.OMITTED_TREE_MARKER.equals(value)) {
      inputMap.putTreeArtifact((SpecialArtifact) treeArtifact, value, depOwner);
      return;
    }

    inputMap.putTreeArtifact((SpecialArtifact) treeArtifact, value, depOwner);
    expandedArtifacts.put(treeArtifact, value.getChildren());

    value
        .getArchivedRepresentation()
        .ifPresent(
            archivedRepresentation -> {
              inputMap.put(
                  archivedRepresentation.archivedTreeFileArtifact(),
                  archivedRepresentation.archivedFileValue(),
                  depOwner);
              archivedTreeArtifacts.put(
                  (SpecialArtifact) treeArtifact,
                  archivedRepresentation.archivedTreeFileArtifact());
            });
  }
}
