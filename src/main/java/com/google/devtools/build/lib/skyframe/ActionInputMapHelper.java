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


import com.google.devtools.build.lib.actions.ActionInputMapSink;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue.ArchivedRepresentation;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.function.BiConsumer;

/** Static utilities for working with action inputs. */
final class ActionInputMapHelper {

  private ActionInputMapHelper() {}

  /** Adds a value obtained by an Artifact SkyValue lookup to the action input map. */
  static void addToMap(
      ActionInputMapSink inputMap,
      BiConsumer<Artifact, TreeArtifactValue> treeArtifactConsumer,
      Map<Artifact, FilesetOutputTree> filesetsInsideRunfiles,
      Map<Artifact, FilesetOutputTree> topLevelFilesets,
      Artifact key,
      SkyValue value,
      MetadataConsumerForMetrics consumer) {
    if (value instanceof RunfilesArtifactValue runfilesArtifactValue) {
      // Note: we don't expand the .runfiles/MANIFEST file into the inputs. The reason for that
      // being that the MANIFEST file contains absolute paths that don't work with remote execution.
      // Instead, the way the SpawnInputExpander expands runfiles is via the Runfiles class
      // which contains all artifacts in the runfiles tree minus the MANIFEST file.
      runfilesArtifactValue.forEachFile(
          (artifact, metadata) -> {
            inputMap.put(artifact, metadata, /* depOwner= */ key);
            consumer.accumulate(metadata);
          });
      runfilesArtifactValue.forEachTree(
          (treeArtifact, metadata) -> {
            expandTreeArtifactAndPopulateArtifactData(
                treeArtifact, metadata, treeArtifactConsumer, inputMap, /* depOwner= */ key);
            consumer.accumulate(metadata);
          });

      runfilesArtifactValue.forEachFileset(
          (fileset, filesetOutputTree) -> {
            filesetsInsideRunfiles.put(fileset, filesetOutputTree);
            consumer.accumulate(filesetOutputTree);
          });
      // This is used for three purposes:
      // - To collect the RunfilesTree objects which the execution strategies need to expand the
      //   runfiles trees
      // - The action cache checker may want the digest of the aggregated runfiles tree
      // - Input rewinding needs to know the owners of artifacts in the runfiles tree
      //   (this is definitely necessary for Filesets; dunno how that works for other artifacts)
      inputMap.putRunfilesMetadata(key, runfilesArtifactValue, /* depOwner= */ key);
    } else if (value instanceof TreeArtifactValue treeArtifactValue) {
      expandTreeArtifactAndPopulateArtifactData(
          key, treeArtifactValue, treeArtifactConsumer, inputMap, /* depOwner= */ key);
      consumer.accumulate(treeArtifactValue);
    } else if (value instanceof ActionExecutionValue actionExecutionValue) {
      // Actions resulting from the expansion of an ActionTemplate consume only one of the files
      // in a tree artifact. However, the input prefetcher and the Linux sandbox require access to
      // the tree metadata to determine the prefetch location of a tree artifact materialized as a
      // symlink to another (cf. TreeArtifactValue#getResolvedPath()).
      if (key.isChildOfDeclaredDirectory()) {
        SpecialArtifact treeArtifact = key.getParent();
        TreeArtifactValue treeArtifactValue =
            actionExecutionValue.getTreeArtifactValue(treeArtifact);
        expandTreeArtifactAndPopulateArtifactData(
            treeArtifact, treeArtifactValue, treeArtifactConsumer, inputMap, treeArtifact);
        consumer.accumulate(treeArtifactValue);
      }
      FileArtifactValue metadata = actionExecutionValue.getExistingFileArtifactValue(key);
      inputMap.put(key, metadata, key);
      if (key.isFileset()) {
        FilesetOutputTree filesetOutput =
            (FilesetOutputTree) actionExecutionValue.getRichArtifactData();
        topLevelFilesets.put(key, filesetOutput);
        consumer.accumulate(filesetOutput);
      } else {
        consumer.accumulate(metadata);
      }
    } else if (value instanceof FileArtifactValue fileArtifactValue) {
      inputMap.put(key, fileArtifactValue, /* depOwner= */ key);
      consumer.accumulate(fileArtifactValue);
    } else {
      throw new IllegalStateException(String.format("Unexpected value %s", value));
    }
  }

  private static void expandTreeArtifactAndPopulateArtifactData(
      Artifact treeArtifact,
      TreeArtifactValue value,
      BiConsumer<Artifact, TreeArtifactValue> treeArtifactConsumer,
      ActionInputMapSink inputMap,
      Artifact depOwner) {
    treeArtifactConsumer.accept(treeArtifact, value);
    inputMap.putTreeArtifact((SpecialArtifact) treeArtifact, value, depOwner);
    if (value.getArchivedRepresentation().isEmpty()) {
      return;
    }

    ArchivedRepresentation archivedRepresentation = value.getArchivedRepresentation().get();
    inputMap.put(
        archivedRepresentation.archivedTreeFileArtifact(),
        archivedRepresentation.archivedFileValue(),
        depOwner);
  }
}
