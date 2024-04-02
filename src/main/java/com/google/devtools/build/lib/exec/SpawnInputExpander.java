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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.MissingExpansionException;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.ForbiddenRelativeSymlinkException;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A helper class for spawn strategies to turn runfiles suppliers into input mappings. This class
 * performs no I/O operations, but only rearranges the files according to how the runfiles should be
 * laid out.
 */
public final class SpawnInputExpander {

  /** Accepts mappings from exec path to {@link ActionInput}. */
  public interface InputSink {
    void acceptMapping(PathFragment execPath, ActionInput input, @Nullable Artifact owner);

    /** Adapts a {@link Map} to a {@link InputSink} which disregards owners. */
    static InputSink fromMap(Map<PathFragment, ActionInput> map) {
      return (execPath, input, owner) -> map.put(execPath, input);
    }
  }

  private final Path execRoot;
  private final RelativeSymlinkBehavior relSymlinkBehavior;
  private final boolean expandArchivedTreeArtifacts;

  public SpawnInputExpander(Path execRoot) {
    this(execRoot, RelativeSymlinkBehavior.ERROR);
  }

  public SpawnInputExpander(Path execRoot, RelativeSymlinkBehavior relSymlinkBehavior) {
    this(execRoot, relSymlinkBehavior, /* expandArchivedTreeArtifacts= */ true);
  }

  public SpawnInputExpander(
      Path execRoot,
      RelativeSymlinkBehavior relSymlinkBehavior,
      boolean expandArchivedTreeArtifacts) {
    this.execRoot = execRoot;
    this.relSymlinkBehavior = relSymlinkBehavior;
    this.expandArchivedTreeArtifacts = expandArchivedTreeArtifacts;
  }

  private static void addMapping(
      InputSink inputSink,
      PathFragment targetLocation,
      ActionInput input,
      PathFragment baseDirectory,
      @Nullable Artifact owner) {
    Preconditions.checkArgument(!targetLocation.isAbsolute(), targetLocation);
    inputSink.acceptMapping(baseDirectory.getRelative(targetLocation), input, owner);
  }

  @VisibleForTesting
  void addSingleRunfilesTreeToInputs(
      RunfilesTree runfilesTree,
      InputSink inputSink,
      ArtifactExpander artifactExpander,
      PathMapper pathMapper,
      PathFragment baseDirectory)
      throws ForbiddenActionInputException {
    addSingleRunfilesTreeToInputs(
        inputSink,
        runfilesTree.getExecPath(),
        runfilesTree.getMapping(),
        artifactExpander,
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
      InputSink inputSink,
      PathFragment root,
      Map<PathFragment, Artifact> mappings,
      ArtifactExpander artifactExpander,
      PathMapper pathMapper,
      PathFragment baseDirectory)
      throws ForbiddenActionInputException {
    Preconditions.checkArgument(!root.isAbsolute(), root);
    for (Map.Entry<PathFragment, Artifact> mapping : mappings.entrySet()) {
      PathFragment location = root.getRelative(mapping.getKey());
      Artifact artifact = mapping.getValue();
      if (artifact == null) {
        addMapping(
            inputSink,
            mapForRunfiles(pathMapper, root, location),
            VirtualActionInput.EMPTY_MARKER,
            baseDirectory,
            /* owner= */ null);
        continue;
      }
      Preconditions.checkArgument(!artifact.isMiddlemanArtifact(), artifact);
      if (artifact.isTreeArtifact()) {
        ArchivedTreeArtifact archivedTreeArtifact =
            expandArchivedTreeArtifacts
                ? null
                : artifactExpander.getArchivedTreeArtifact((SpecialArtifact) artifact);
        if (archivedTreeArtifact != null) {
          // TODO(bazel-team): Add path mapping support for archived tree artifacts.
          addMapping(inputSink, location, archivedTreeArtifact, baseDirectory, artifact);
        } else {
          List<ActionInput> expandedInputs =
              ActionInputHelper.expandArtifacts(
                  NestedSetBuilder.create(Order.STABLE_ORDER, artifact),
                  artifactExpander,
                  /* keepEmptyTreeArtifacts= */ false,
                  /* keepMiddlemanArtifacts= */ false);
          for (ActionInput input : expandedInputs) {
            addMapping(
                inputSink,
                mapForRunfiles(pathMapper, root, location)
                    .getRelative(((TreeFileArtifact) input).getParentRelativePath()),
                input,
                baseDirectory,
                artifact);
          }
        }
      } else if (artifact.isFileset()) {
        ImmutableList<FilesetOutputSymlink> filesetLinks;
        try {
          filesetLinks = artifactExpander.getFileset(artifact);
        } catch (MissingExpansionException e) {
          throw new IllegalStateException(e);
        }
        // TODO(bazel-team): Add path mapping support for filesets.
        addFilesetManifest(location, artifact, filesetLinks, inputSink, baseDirectory);
      } else {
        // TODO: b/7075837 - If we want to prohibit directory inputs, we can check if
        //  localArtifact is a directory and, if so, throw a ForbiddenActionInputException.
        addMapping(
            inputSink,
            mapForRunfiles(pathMapper, root, location),
            artifact,
            baseDirectory,
            /* owner= */ null);
      }
    }
  }

  @VisibleForTesting
  void addFilesetManifests(
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings,
      InputSink inputSink,
      PathFragment baseDirectory)
      throws ForbiddenRelativeSymlinkException {
    for (Map.Entry<Artifact, ImmutableList<FilesetOutputSymlink>> entry :
        filesetMappings.entrySet()) {
      Artifact fileset = entry.getKey();
      addFilesetManifest(
          fileset.getExecPath(), fileset, entry.getValue(), inputSink, baseDirectory);
    }
  }

  private void addFilesetManifest(
      PathFragment location,
      Artifact filesetArtifact,
      ImmutableList<FilesetOutputSymlink> filesetLinks,
      InputSink inputSink,
      PathFragment baseDirectory)
      throws ForbiddenRelativeSymlinkException {
    Preconditions.checkArgument(filesetArtifact.isFileset(), filesetArtifact);
    FilesetManifest filesetManifest =
        FilesetManifest.constructFilesetManifest(filesetLinks, location, relSymlinkBehavior);

    for (Map.Entry<PathFragment, String> mapping : filesetManifest.getEntries().entrySet()) {
      String value = mapping.getValue();
      ActionInput artifact =
          value == null
              ? VirtualActionInput.EMPTY_MARKER
              : ActionInputHelper.fromPath(execRoot.getRelative(value).asFragment());
      // TODO(bazel-team): Add path mapping support for filesets.
      addMapping(inputSink, mapping.getKey(), artifact, baseDirectory, filesetArtifact);
    }
  }

  private void addInputs(
      Map<PathFragment, ActionInput> inputMap,
      NestedSet<? extends ActionInput> inputFiles,
      ArtifactExpander artifactExpander,
      InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper,
      PathFragment baseDirectory)
      throws ForbiddenActionInputException {
    InputSink inputSink = InputSink.fromMap(inputMap);
    // Actions that accept TreeArtifacts as inputs generally expect the directory corresponding
    // to the artifact to be created, even if it is empty. We explicitly keep empty TreeArtifacts
    // here to signal consumers that they should create the directory.
    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(
            inputFiles,
            artifactExpander,
            /* keepEmptyTreeArtifacts= */ true,
            /* keepMiddlemanArtifacts= */ true);
    for (ActionInput input : inputs) {
      if (input instanceof TreeFileArtifact) {
        addMapping(
            inputSink,
            pathMapper
                .map(((TreeFileArtifact) input).getParent().getExecPath())
                .getRelative(((TreeFileArtifact) input).getParentRelativePath()),
            input,
            baseDirectory,
            // Owners are disregarded since we're aggregating into a map.
            /* owner= */ null);
      } else if (isMiddlemanArtifact(input)) {
        RunfilesTree runfilesTree =
            inputMetadataProvider.getRunfilesMetadata(input).getRunfilesTree();
        addSingleRunfilesTreeToInputs(
            runfilesTree, inputSink, artifactExpander, pathMapper, baseDirectory);
      } else {
        addMapping(
            inputSink,
            pathMapper.map(input.getExecPath()),
            input,
            baseDirectory,
            // Owners are disregarded since we're aggregating into a map.
            /* owner= */ null);
      }
    }
  }

  /**
   * Convert the inputs and runfiles of the given spawn to a map from exec-root relative paths to
   * {@link ActionInput}s. The returned map does not contain non-empty tree artifacts as they are
   * expanded to file artifacts. Tree artifacts that would expand to the empty set under the
   * provided {@link ArtifactExpander} are left untouched so that their corresponding empty
   * directories can be created.
   *
   * <p>The returned map never contains {@code null} values.
   *
   * <p>The returned map contains all runfiles, but not the {@code MANIFEST}.
   */
  public SortedMap<PathFragment, ActionInput> getInputMapping(
      Spawn spawn,
      ArtifactExpander artifactExpander,
      InputMetadataProvider inputMetadataProvider,
      PathFragment baseDirectory)
      throws ForbiddenActionInputException {
    TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
    InputSink inputSink = InputSink.fromMap(inputMap);
    addInputs(
        inputMap,
        spawn.getInputFiles(),
        artifactExpander,
        inputMetadataProvider,
        spawn.getPathMapper(),
        baseDirectory);
    addFilesetManifests(spawn.getFilesetMappings(), inputSink, baseDirectory);
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

  /** The interface for accessing part of the input hierarchy. */
  public interface InputWalker {

    /** Returns the leaf nodes at this point in the hierarchy. */
    SortedMap<PathFragment, ActionInput> getLeavesInputMapping()
        throws IOException, ForbiddenActionInputException;

    /** Invokes the visitor on the non-leaf nodes at this point in the hierarchy. */
    default void visitNonLeaves(InputVisitor visitor)
        throws IOException, ForbiddenActionInputException {}
  }

  /** The interface for visiting part of the input hierarchy. */
  public interface InputVisitor {

    /**
     * Visits a part of the input hierarchy.
     *
     * <p>{@code nodeKey} can be used as key when memoizing visited parts of the hierarchy.
     */
    void visit(Object nodeKey, InputWalker walker)
        throws IOException, ForbiddenActionInputException;
  }

  /**
   * Visits the input files hierarchy in a depth first manner.
   *
   * <p>Similar to {@link #getInputMapping} but allows for early exit, by not visiting children,
   * when walking through the input hierarchy. By applying memoization, the retrieval process of the
   * inputs can be speeded up.
   *
   * <p>{@code baseDirectory} is prepended to every path in the input key. This is useful if the
   * mapping is used in a context where the directory relative to which the keys are interpreted is
   * not the same as the execroot.
   */
  public void walkInputs(
      Spawn spawn,
      ArtifactExpander artifactExpander,
      InputMetadataProvider inputMetadataProvider,
      PathFragment baseDirectory,
      InputVisitor visitor)
      throws IOException, ForbiddenActionInputException {
    walkNestedSetInputs(
        baseDirectory,
        spawn.getInputFiles(),
        artifactExpander,
        inputMetadataProvider,
        spawn.getPathMapper(),
        visitor);

    Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings = spawn.getFilesetMappings();
    // filesetMappings is assumed to be very small, so no need to implement visitNonLeaves() for
    // improved runtime.
    visitor.visit(
        // Cache key for the sub-mapping containing the fileset inputs for this spawn.
        ImmutableList.of(filesetMappings, baseDirectory, spawn.getPathMapper().cacheKey()),
        new InputWalker() {
          @Override
          public SortedMap<PathFragment, ActionInput> getLeavesInputMapping()
              throws ForbiddenRelativeSymlinkException {
            TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
            addFilesetManifests(filesetMappings, InputSink.fromMap(inputMap), baseDirectory);
            return inputMap;
          }
        });
  }

  /** Visits a {@link NestedSet} occurring in {@link Spawn#getInputFiles}. */
  private void walkNestedSetInputs(
      PathFragment baseDirectory,
      NestedSet<? extends ActionInput> someInputFiles,
      ArtifactExpander artifactExpander,
      InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper,
      InputVisitor visitor)
      throws IOException, ForbiddenActionInputException {
    visitor.visit(
        // Cache key for the sub-mapping containing the files in this nested set.
        ImmutableList.of(someInputFiles.toNode(), baseDirectory, pathMapper.cacheKey()),
        new InputWalker() {
          @Override
          public SortedMap<PathFragment, ActionInput> getLeavesInputMapping()
              throws ForbiddenActionInputException {
            TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
            // Consider files inside tree artifacts and runfiles trees to be non-leaves. This caches
            // better when a large tree is not the sole direct child of a nested set.
            ImmutableList<? extends ActionInput> leaves =
                someInputFiles.getLeaves().stream()
                    .filter(a -> !isTreeArtifact(a) && !isMiddlemanArtifact(a))
                    .collect(toImmutableList());
            addInputs(
                inputMap,
                NestedSetBuilder.wrap(someInputFiles.getOrder(), leaves),
                artifactExpander,
                inputMetadataProvider,
                pathMapper,
                baseDirectory);
            return inputMap;
          }

          @Override
          public void visitNonLeaves(InputVisitor childVisitor)
              throws IOException, ForbiddenActionInputException {
            for (ActionInput input : someInputFiles.getLeaves()) {
              if (isTreeArtifact(input)) {
                walkTreeInputs(
                    baseDirectory,
                    (SpecialArtifact) input,
                    artifactExpander,
                    inputMetadataProvider,
                    pathMapper,
                    childVisitor);
              }

              if (isMiddlemanArtifact(input)) {
                walkRunfilesTree(
                    baseDirectory,
                    inputMetadataProvider.getRunfilesMetadata(input).getRunfilesTree(),
                    artifactExpander,
                    pathMapper,
                    childVisitor);
              }
            }

            for (NestedSet<? extends ActionInput> subInputs : someInputFiles.getNonLeaves()) {
              walkNestedSetInputs(
                  baseDirectory,
                  subInputs,
                  artifactExpander,
                  inputMetadataProvider,
                  pathMapper,
                  childVisitor);
            }
          }
        });
  }

  private void walkRunfilesTree(
      PathFragment baseDirectory,
      RunfilesTree runfilesTree,
      ArtifactExpander artifactExpander,
      PathMapper pathMapper,
      InputVisitor visitor)
      throws IOException, ForbiddenActionInputException {
    visitor.visit(
        // Cache key for the sub-mapping containing this runfiles tree.
        ImmutableList.of(runfilesTree.getExecPath(), baseDirectory, pathMapper.cacheKey()),
        new InputWalker() {
          @Override
          public SortedMap<PathFragment, ActionInput> getLeavesInputMapping()
              throws ForbiddenActionInputException {
            TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
            addSingleRunfilesTreeToInputs(
                runfilesTree,
                InputSink.fromMap(inputMap),
                artifactExpander,
                pathMapper,
                baseDirectory);
            return inputMap;
          }
        });
  }

  /** Visits a tree artifact occurring in {@link Spawn#getInputFiles}. */
  private void walkTreeInputs(
      PathFragment baseDirectory,
      SpecialArtifact tree,
      ArtifactExpander artifactExpander,
      InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper,
      InputVisitor visitor)
      throws IOException, ForbiddenActionInputException {
    visitor.visit(
        // Cache key for the sub-mapping containing the files in this tree artifact.
        ImmutableList.of(tree, baseDirectory, pathMapper.cacheKey()),
        new InputWalker() {
          @Override
          public SortedMap<PathFragment, ActionInput> getLeavesInputMapping()
              throws ForbiddenActionInputException {
            TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
            addInputs(
                inputMap,
                NestedSetBuilder.create(Order.STABLE_ORDER, tree),
                artifactExpander,
                inputMetadataProvider,
                pathMapper,
                baseDirectory);
            return inputMap;
          }
        });
  }

  private static boolean isTreeArtifact(ActionInput input) {
    return input instanceof SpecialArtifact && ((SpecialArtifact) input).isTreeArtifact();
  }

  private static boolean isMiddlemanArtifact(ActionInput input) {
    return input instanceof Artifact && ((Artifact) input).isMiddlemanArtifact();
  }
}
