// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * A helper class for spawn strategies to turn runfiles suppliers into input mappings. This class
 * performs no I/O operations, but only rearranges the files according to how the runfiles should be
 * laid out.
 */
public final class SpawnInputExpander {

  private final boolean expandArchivedTreeArtifacts;

  public SpawnInputExpander() {
    this(/* expandArchivedTreeArtifacts= */ true);
  }

  public SpawnInputExpander(boolean expandArchivedTreeArtifacts) {
    this.expandArchivedTreeArtifacts = expandArchivedTreeArtifacts;
  }

  private static void addMapping(
      Map<PathFragment, ActionInput> inputMap,
      PathFragment targetLocation,
      ActionInput input,
      PathFragment baseDirectory) {
    Preconditions.checkArgument(!targetLocation.isAbsolute(), targetLocation);
    inputMap.put(baseDirectory.getRelative(targetLocation), input);
  }

  @VisibleForTesting
  void addSingleRunfilesTreeToInputs(
      RunfilesTree runfilesTree,
      Map<PathFragment, ActionInput> inputMap,
      InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper,
      PathFragment baseDirectory) {
    addSingleRunfilesTreeToInputs(
        inputMap,
        runfilesTree.getExecPath(),
        runfilesTree.getMapping(),
        inputMetadataProvider,
        pathMapper,
        baseDirectory);
  }

  /**
   * Gathers the mapping for a single runfiles tree into {@code inputMap}.
   *
   * <p>This should not be a public interface, it's only there to support legacy code until we
   * figure out how not to call this method (or else how to make this method more palatable)
   */
  public void addSingleRunfilesTreeToInputs(
      Map<PathFragment, ActionInput> inputMap,
      PathFragment root,
      Map<PathFragment, Artifact> mappings,
      InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper,
      PathFragment baseDirectory) {
    Preconditions.checkArgument(!root.isAbsolute(), root);
    for (Map.Entry<PathFragment, Artifact> mapping : mappings.entrySet()) {
      PathFragment location = root.getRelative(mapping.getKey());
      Artifact artifact = mapping.getValue();
      if (artifact == null) {
        addMapping(
            inputMap,
            mapForRunfiles(pathMapper, root, location),
            VirtualActionInput.EMPTY_MARKER,
            baseDirectory);
        continue;
      }
      Preconditions.checkArgument(!artifact.isRunfilesTree(), artifact);
      if (artifact.isTreeArtifact()) {
        TreeArtifactValue treeArtifactValue = inputMetadataProvider.getTreeMetadata(artifact);
        ArchivedTreeArtifact archivedTreeArtifact =
            expandArchivedTreeArtifacts ? null : treeArtifactValue.getArchivedArtifact();
        if (archivedTreeArtifact != null) {
          // TODO(bazel-team): Add path mapping support for archived tree artifacts.
          addMapping(inputMap, location, archivedTreeArtifact, baseDirectory);
        } else if (treeArtifactValue.getChildren().isEmpty()) {
          addMapping(inputMap, mapForRunfiles(pathMapper, root, location), artifact, baseDirectory);
        } else {
          for (TreeFileArtifact input : treeArtifactValue.getChildren()) {
            addMapping(
                inputMap,
                mapForRunfiles(pathMapper, root, location)
                    .getRelative(input.getParentRelativePath()),
                input,
                baseDirectory);
          }
        }
      } else if (artifact.isFileset()) {
        // TODO(bazel-team): Add path mapping support for filesets.
        FilesetOutputTree filesetOutput = inputMetadataProvider.getFileset(artifact);
        addFilesetManifest(location, artifact, filesetOutput, inputMap, baseDirectory);
      } else {
        addMapping(inputMap, mapForRunfiles(pathMapper, root, location), artifact, baseDirectory);
      }
    }
  }

  @VisibleForTesting
  static void addFilesetManifests(
      Map<Artifact, FilesetOutputTree> filesetMappings,
      Map<PathFragment, ActionInput> inputMap,
      PathFragment baseDirectory) {
    for (Map.Entry<Artifact, FilesetOutputTree> entry : filesetMappings.entrySet()) {
      Artifact fileset = entry.getKey();
      addFilesetManifest(fileset.getExecPath(), fileset, entry.getValue(), inputMap, baseDirectory);
    }
  }

  private static void addFilesetManifest(
      PathFragment location,
      Artifact filesetArtifact,
      FilesetOutputTree filesetOutput,
      Map<PathFragment, ActionInput> inputMap,
      PathFragment baseDirectory) {
    Preconditions.checkArgument(filesetArtifact.isFileset(), filesetArtifact);
    for (FilesetOutputSymlink link : filesetOutput.symlinks()) {
      addMapping(inputMap, location.getRelative(link.name()), link.target(), baseDirectory);
    }
  }

  private void addInputs(
      Map<PathFragment, ActionInput> inputMap,
      NestedSet<? extends ActionInput> inputFiles,
      InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper,
      PathFragment baseDirectory) {
    // Actions that accept TreeArtifacts as inputs generally expect the directory corresponding
    // to the artifact to be created, even if it is empty. We explicitly keep empty TreeArtifacts
    // here to signal consumers that they should create the directory.
    List<ActionInput> inputs =
        InputMetadataProvider.expandArtifacts(
            inputMetadataProvider,
            inputFiles,
            /* keepEmptyTreeArtifacts= */ true,
            /* keepRunfilesTrees= */ true);
    for (ActionInput input : inputs) {
      switch (input) {
        case TreeFileArtifact treeFileArtifact ->
            addMapping(
                inputMap,
                pathMapper
                    .map(treeFileArtifact.getParent().getExecPath())
                    .getRelative(treeFileArtifact.getParentRelativePath()),
                input,
                baseDirectory);
        case Artifact runfilesTreeArtifact when runfilesTreeArtifact.isRunfilesTree() ->
            addSingleRunfilesTreeToInputs(
                inputMetadataProvider.getRunfilesMetadata(runfilesTreeArtifact).getRunfilesTree(),
                inputMap,
                inputMetadataProvider,
                pathMapper,
                baseDirectory);
        case Artifact fileset when fileset.isFileset() ->
            addFilesetManifest(
                fileset.getExecPath(),
                fileset,
                inputMetadataProvider.getFileset(fileset),
                inputMap,
                baseDirectory);
        default -> addMapping(inputMap, pathMapper.map(input.getExecPath()), input, baseDirectory);
      }
    }
  }

  /**
   * Convert the inputs and runfiles of the given spawn to a map from exec-root relative paths to
   * {@link ActionInput}s. The returned map does not contain non-empty tree artifacts as they are
   * expanded to file artifacts. Tree artifacts that would expand to the empty set under the
   * provided {@link InputMetadataProvider} are left untouched so that their corresponding empty
   * directories can be created.
   *
   * <p>The returned map never contains {@code null} values.
   *
   * <p>The returned map contains all runfiles, but not the {@code MANIFEST}.
   */
  public SortedMap<PathFragment, ActionInput> getInputMapping(
      Spawn spawn, InputMetadataProvider inputMetadataProvider, PathFragment baseDirectory) {
    TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
    addInputs(
        inputMap,
        spawn.getInputFiles(),
        inputMetadataProvider,
        spawn.getPathMapper(),
        baseDirectory);
    return inputMap;
  }

  private static PathFragment mapForRunfiles(
      PathMapper pathMapper, PathFragment runfilesDir, PathFragment execPath) {
    if (pathMapper.isNoop()) {
      return execPath;
    }
    String runfilesDirName = runfilesDir.getBaseName();
    Preconditions.checkArgument(runfilesDirName.endsWith(".runfiles"));
    // Derive the path of the executable, apply the path mapping to it and then rederive the path
    // of the runfiles dir.
    PathFragment executable =
        runfilesDir.replaceName(
            runfilesDirName.substring(0, runfilesDirName.length() - ".runfiles".length()));
    return pathMapper
        .map(executable)
        .replaceName(runfilesDirName)
        .getRelative(execPath.relativeTo(runfilesDir));
  }
}
