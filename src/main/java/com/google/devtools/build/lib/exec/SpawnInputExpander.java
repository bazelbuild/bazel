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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput.EmptyActionInput;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * A helper class for spawn strategies to turn runfiles suppliers into input mappings. This class
 * performs no I/O operations, but only rearranges the files according to how the runfiles should be
 * laid out.
 */
public class SpawnInputExpander {
  public static final ActionInput EMPTY_FILE = new EmptyActionInput("/dev/null");

  private final Path execRoot;
  private final boolean strict;
  private final RelativeSymlinkBehavior relSymlinkBehavior;

  /**
   * Creates a new instance. If strict is true, then the expander checks for directories in runfiles
   * and throws an exception if it finds any. Otherwise it silently ignores directories in runfiles
   * and adds a mapping for them. At this time, directories in filesets are always silently added as
   * mappings.
   *
   * <p>Directories in inputs are a correctness issue: Bazel only tracks dependencies at the action
   * level, and it does not track dependencies on directories. Making a directory available to a
   * spawn even though it's contents are not tracked as dependencies leads to incorrect incremental
   * builds, since changes to the contents do not trigger action invalidation.
   *
   * <p>As such, all spawn strategies should always be strict and not make directories available to
   * the subprocess. However, that's a breaking change, and therefore we make it depend on this flag
   * for now.
   */
  public SpawnInputExpander(Path execRoot, boolean strict) {
    this(execRoot, strict, RelativeSymlinkBehavior.ERROR);
  }

  /**
   * Creates a new instance. If strict is true, then the expander checks for directories in runfiles
   * and throws an exception if it finds any. Otherwise it silently ignores directories in runfiles
   * and adds a mapping for them. At this time, directories in filesets are always silently added as
   * mappings.
   *
   * <p>Directories in inputs are a correctness issue: Bazel only tracks dependencies at the action
   * level, and it does not track dependencies on directories. Making a directory available to a
   * spawn even though it's contents are not tracked as dependencies leads to incorrect incremental
   * builds, since changes to the contents do not trigger action invalidation.
   *
   * <p>As such, all spawn strategies should always be strict and not make directories available to
   * the subprocess. However, that's a breaking change, and therefore we make it depend on this flag
   * for now.
   */
  public SpawnInputExpander(
      Path execRoot, boolean strict, RelativeSymlinkBehavior relSymlinkBehavior) {
    this.execRoot = execRoot;
    this.strict = strict;
    this.relSymlinkBehavior = relSymlinkBehavior;
  }

  private void addMapping(
      Map<PathFragment, ActionInput> inputMappings,
      PathFragment targetLocation,
      ActionInput input) {
    Preconditions.checkArgument(!targetLocation.isAbsolute(), targetLocation);
    inputMappings.put(targetLocation, input);
  }

  /** Adds runfiles inputs from runfilesSupplier to inputMappings. */
  @VisibleForTesting
  void addRunfilesToInputs(
      Map<PathFragment, ActionInput> inputMap,
      RunfilesSupplier runfilesSupplier,
      MetadataProvider actionFileCache,
      ArtifactExpander artifactExpander)
      throws IOException {
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        runfilesSupplier.getMappings();

    for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root = rootAndMappings.getKey();
      Preconditions.checkState(!root.isAbsolute(), root);
      for (Map.Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        PathFragment location = root.getRelative(mapping.getKey());
        Artifact localArtifact = mapping.getValue();
        if (localArtifact != null) {
          Preconditions.checkState(!localArtifact.isMiddlemanArtifact());
          if (localArtifact.isTreeArtifact()) {
            List<ActionInput> expandedInputs =
                ActionInputHelper.expandArtifacts(
                    NestedSetBuilder.create(Order.STABLE_ORDER, localArtifact), artifactExpander);
            for (ActionInput input : expandedInputs) {
              addMapping(
                  inputMap,
                  location.getRelative(((TreeFileArtifact) input).getParentRelativePath()),
                  input);
            }
          } else if (localArtifact.isFileset()) {
            addFilesetManifest(
                location, localArtifact, artifactExpander.getFileset(localArtifact), inputMap);
          } else {
            if (strict) {
              failIfDirectory(actionFileCache, localArtifact);
            }
            addMapping(inputMap, location, localArtifact);
          }
        } else {
          addMapping(inputMap, location, EMPTY_FILE);
        }
      }
    }
  }

  /** Adds runfiles inputs from runfilesSupplier to inputMappings. */
  public Map<PathFragment, ActionInput> addRunfilesToInputs(
      RunfilesSupplier runfilesSupplier,
      MetadataProvider actionFileCache,
      ArtifactExpander artifactExpander)
      throws IOException {
    Map<PathFragment, ActionInput> inputMap = new HashMap<>();
    addRunfilesToInputs(inputMap, runfilesSupplier, actionFileCache, artifactExpander);
    return inputMap;
  }

  private static void failIfDirectory(MetadataProvider actionFileCache, ActionInput input)
      throws IOException {
    FileArtifactValue metadata = actionFileCache.getMetadata(input);
    if (metadata != null && !metadata.getType().isFile()) {
      throw new IOException("Not a file: " + input.getExecPathString());
    }
  }

  @VisibleForTesting
  void addFilesetManifests(
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings,
      Map<PathFragment, ActionInput> inputMappings)
      throws IOException {
    for (Artifact fileset : filesetMappings.keySet()) {
      addFilesetManifest(
          fileset.getExecPath(), fileset, filesetMappings.get(fileset), inputMappings);
    }
  }

  void addFilesetManifest(
      PathFragment location,
      Artifact filesetArtifact,
      ImmutableList<FilesetOutputSymlink> filesetLinks,
      Map<PathFragment, ActionInput> inputMappings)
      throws IOException {
    Preconditions.checkState(filesetArtifact.isFileset(), filesetArtifact);
    FilesetManifest filesetManifest =
        FilesetManifest.constructFilesetManifest(filesetLinks, location, relSymlinkBehavior);

      for (Map.Entry<PathFragment, String> mapping : filesetManifest.getEntries().entrySet()) {
        String value = mapping.getValue();
        ActionInput artifact =
            value == null
                ? EMPTY_FILE
                : ActionInputHelper.fromPath(execRoot.getRelative(value).getPathString());
        addMapping(inputMappings, mapping.getKey(), artifact);
      }
  }

  private void addInputs(
      Map<PathFragment, ActionInput> inputMap, Spawn spawn, ArtifactExpander artifactExpander) {
    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(spawn.getInputFiles(), artifactExpander);
    for (ActionInput input : inputs) {
      addMapping(inputMap, input.getExecPath(), input);
    }
  }

  /**
   * Convert the inputs and runfiles of the given spawn to a map from exec-root relative paths to
   * {@link ActionInput}s. The returned map does not contain tree artifacts as they are expanded to
   * file artifacts.
   *
   * <p>The returned map never contains {@code null} values; it uses {@link #EMPTY_FILE} for empty
   * files, which is an instance of {@link
   * com.google.devtools.build.lib.actions.cache.VirtualActionInput}.
   *
   * <p>The returned map contains all runfiles, but not the {@code MANIFEST}.
   */
  public SortedMap<PathFragment, ActionInput> getInputMapping(
      Spawn spawn, ArtifactExpander artifactExpander, MetadataProvider actionInputFileCache)
      throws IOException {

    TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
    addInputs(inputMap, spawn, artifactExpander);
    addRunfilesToInputs(
        inputMap, spawn.getRunfilesSupplier(), actionInputFileCache, artifactExpander);
    addFilesetManifests(spawn.getFilesetMappings(), inputMap);
    return inputMap;
  }
}
