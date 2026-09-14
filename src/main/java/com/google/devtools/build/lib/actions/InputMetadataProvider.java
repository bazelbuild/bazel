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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RewindableRepoFileSystem;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import javax.annotation.Nullable;

/** Provides {@link ActionInput} metadata. */
@ThreadSafe
public interface InputMetadataProvider {
  /**
   * Returns a {@link FileArtifactValue} for the given {@link ActionInput}.
   *
   * <p>The returned {@link FileArtifactValue} instance corresponds to the final target of a symlink
   * and therefore must not have a type of {@link FileStateType#SYMLINK}.
   *
   * <p>If {@linkplain DerivedArtifact#getGeneratingActionKey generating action} is not immediately
   * available, this method throws {@code MissingDepExecException} to signal that a Skyframe restart
   * is necessary to obtain the requested metadata.
   *
   * @param input the input to retrieve the digest for
   * @return the artifact's digest or null if digest cannot be obtained (due to artifact
   *     non-existence, lookup errors, or any other reason)
   * @throws InterruptedException if interrupted
   * @throws IOException if the action input cannot be digested
   * @throws MissingDepExecException if a Skyframe restart is required to provide the requested data
   */
  @Nullable
  FileArtifactValue getInputMetadataChecked(ActionInput input)
      throws InterruptedException, IOException, MissingDepExecException;

  /**
   * Returns the {@link TreeArtifactValue} for the given path, or {@code null} if no such tree
   * artifact exists.
   */
  @Nullable
  TreeArtifactValue getTreeMetadata(ActionInput input);

  /**
   * Returns the {@link TreeArtifactValue} for the tree artifact that contains the given path or
   * {@code null} if no such tree artifact exists.
   */
  @Nullable
  TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath);

  /**
   * Like {@link #getInputMetadata(ActionInput)}, but assumes that no Skyframe restart is needed.
   *
   * <p>If one is needed anyway, throws {@link IllegalStateException}.
   */
  @Nullable
  default FileArtifactValue getInputMetadata(ActionInput input) throws IOException {
    try {
      return getInputMetadataChecked(input);
    } catch (MissingDepExecException | InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the contents of a given Fileset on the inputs of the action.
   *
   * <p>Works both for Filesets that are directly on the inputs and those that are included in a
   * runfiles tree.
   */
  @Nullable
  FilesetOutputTree getFileset(ActionInput input);

  /**
   * Returns the Filesets on the inputs of the action.
   *
   * <p>Contains both Filesets that are directly on the inputs and those that are included in a
   * runfiles tree.
   */
  Map<Artifact, FilesetOutputTree> getFilesets();

  /**
   * Returns the {@link RunfilesArtifactValue} for the given {@link ActionInput}, which must be a
   * runfiles tree artifact.
   *
   * @return the appropriate {@link RunfilesArtifactValue} or null if it's not found.
   */
  @Nullable
  RunfilesArtifactValue getRunfilesMetadata(ActionInput input);

  /** Returns the runfiles trees in this metadata provider. */
  ImmutableList<RunfilesTree> getRunfilesTrees();

  /**
   * Returns the repositories whose contents are read by reading the given inputs and the runfiles
   * trees: those containing external source artifacts as well as those that derived artifacts
   * resolve into, such as the output of a symlink action whose target is an external source file
   * (see {@link FileArtifactValue#getResolvedPath}).
   */
  default ImmutableSet<RepositoryName> getExternalSourceRepositories(
      Iterable<Artifact> inputs, RewindableRepoFileSystem repoFileSystem) {
    // Roots are interned, so names are only looked up once per distinct external source root.
    Set<ArtifactRoot> roots = new HashSet<>();
    Set<RepositoryName> repos = new HashSet<>();
    collectExternalSourceRepositories(inputs, repoFileSystem, roots, repos);
    for (RunfilesTree runfilesTree : getRunfilesTrees()) {
      collectExternalSourceRepositories(
          runfilesTree.getArtifacts().toList(), repoFileSystem, roots, repos);
    }
    for (ArtifactRoot root : roots) {
      repos.add(root.getExternalRepositoryName());
    }
    return ImmutableSet.copyOf(repos);
  }

  private void collectExternalSourceRepositories(
      Iterable<Artifact> artifacts,
      RewindableRepoFileSystem repoFileSystem,
      Set<ArtifactRoot> roots,
      Set<RepositoryName> repos) {
    for (Artifact artifact : artifacts) {
      if (artifact.isSourceArtifact()) {
        if (artifact.getRoot().getRootType() == ArtifactRoot.RootType.EXTERNAL_SOURCE) {
          roots.add(artifact.getRoot());
        }
      } else if (artifact.isTreeArtifact()) {
        TreeArtifactValue tree = getTreeMetadata(artifact);
        if (tree != null) {
          addRepositoryOfResolvedPath(tree.getMetadata(), repoFileSystem, repos);
          for (FileArtifactValue child : tree.getChildValues().values()) {
            addRepositoryOfResolvedPath(child, repoFileSystem, repos);
          }
        }
      } else {
        FileArtifactValue metadata;
        try {
          metadata = getInputMetadata(artifact);
        } catch (IOException e) {
          // The metadata of inputs is computed before they are read, so this doesn't do I/O.
          throw new IllegalStateException(e);
        }
        if (metadata != null) {
          addRepositoryOfResolvedPath(metadata, repoFileSystem, repos);
        }
      }
    }
  }

  private static void addRepositoryOfResolvedPath(
      FileArtifactValue metadata,
      RewindableRepoFileSystem repoFileSystem,
      Set<RepositoryName> repos) {
    PathFragment resolvedPath = metadata.getResolvedPath();
    if (resolvedPath != null && repoFileSystem.isRepoPath(resolvedPath)) {
      repos.add(repoFileSystem.repoContaining(resolvedPath));
    }
  }

  /** Looks up an input from its exec path. */
  @Nullable
  ActionInput getInput(PathFragment execPath);

  /**
   * Expands tree artifacts in a sequence of {@link ActionInput}s.
   *
   * <p>If {@code keepEmptyTreeArtifacts} is true, a tree artifact will be included in the
   * constructed list when it expands into zero file artifacts. Otherwise, only the file artifacts
   * the tree artifact expands into will be included.
   *
   * <p>Runfiles tree artifacts will be returned if {@code keepRunfilesTrees} is set, otherwise they
   * will be filtered out.
   *
   * <p>Non-runfiles, non-tree artifacts are returned untouched.
   */
  static List<ActionInput> expandArtifacts(
      InputMetadataProvider inputMetadataProvider,
      Iterable<? extends ActionInput> inputs,
      boolean keepEmptyTreeArtifacts,
      boolean keepRunfilesTrees) {
    List<ActionInput> result = new ArrayList<>();
    Set<Artifact> emptyTreeArtifacts = new TreeSet<>();
    Set<Artifact> treeFileArtifactParents = new HashSet<>();
    for (ActionInput input : inputs) {
      if (!(input instanceof Artifact artifact)) {
        result.add(input);
      } else if (artifact.isRunfilesTree()) {
        if (keepRunfilesTrees) {
          result.add(artifact);
        }
      } else if (artifact.isTreeArtifact()) {
        TreeArtifactValue treeArtifactValue = inputMetadataProvider.getTreeMetadata(artifact);
        if (treeArtifactValue == null || treeArtifactValue.getChildren().isEmpty()) {
          emptyTreeArtifacts.add(artifact);
        } else {
          result.addAll(treeArtifactValue.getChildren());
        }
      } else {
        result.add(artifact);
        if (artifact.isChildOfDeclaredDirectory()) {
          treeFileArtifactParents.add(artifact.getParent());
        }
      }
    }

    if (keepEmptyTreeArtifacts) {
      emptyTreeArtifacts.removeAll(treeFileArtifactParents);
      result.addAll(emptyTreeArtifacts);
    }
    return result;
  }
}
