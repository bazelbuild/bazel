// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehaviorWithoutError;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Container for the data one needs to resolve aggregate artifacts from events signaling the
 * completion of a target or an aspect ({@code TargetCompleteEvent} and {@code
 * AspectCompleteEvent}).
 *
 * <p>This is needed because some artifacts (tree artifacts and Filesets) are in fact aggregations
 * of multiple files.
 */
public final class CompletionContext implements ArtifactExpander {
  public static final CompletionContext FAILED_COMPLETION_CTX =
      new CompletionContext(
          null,
          ImmutableMap.of(),
          ImmutableMap.of(),
          ArtifactPathResolver.IDENTITY,
          new ActionInputMap(BugReporter.defaultInstance(), 0),
          false,
          false);

  private final Path execRoot;
  private final ArtifactPathResolver pathResolver;
  private final Map<Artifact, ImmutableSortedSet<TreeFileArtifact>> expandedTreeArtifacts;
  private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets;
  // Only contains the metadata for 'important' artifacts of the Target/Aspect that completed. Any
  // 'unimportant' artifacts produced by internal output groups (most importantly, _validation) will
  // not be included to avoid retaining many GB on the heap. This ActionInputMap must only be
  // consulted with respect to known-important artifacts (e.g. artifacts referenced in BEP).
  private final ActionInputMap importantInputMap;
  private final boolean expandFilesets;
  private final boolean fullyResolveFilesetLinks;

  @VisibleForTesting
  public CompletionContext(
      Path execRoot,
      Map<Artifact, ImmutableSortedSet<TreeFileArtifact>> expandedTreeArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      ArtifactPathResolver pathResolver,
      ActionInputMap importantInputMap,
      boolean expandFilesets,
      boolean fullyResolveFilesetLinks) {
    this.execRoot = execRoot;
    this.expandedTreeArtifacts = expandedTreeArtifacts;
    this.expandedFilesets = expandedFilesets;
    this.pathResolver = pathResolver;
    this.importantInputMap = importantInputMap;
    this.expandFilesets = expandFilesets;
    this.fullyResolveFilesetLinks = fullyResolveFilesetLinks;
  }

  public static CompletionContext create(
      Map<Artifact, ImmutableSortedSet<TreeFileArtifact>> expandedTreeArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      boolean expandFilesets,
      boolean fullyResolveFilesetSymlinks,
      ActionInputMap inputMap,
      ActionInputMap importantInputMap,
      PathResolverFactory pathResolverFactory,
      Path execRoot,
      String workspaceName) {
    ArtifactPathResolver pathResolver =
        pathResolverFactory.shouldCreatePathResolverForArtifactValues()
            ? pathResolverFactory.createPathResolverForArtifactValues(
                inputMap, expandedTreeArtifacts, expandedFilesets, workspaceName)
            : ArtifactPathResolver.IDENTITY;
    return new CompletionContext(
        execRoot,
        expandedTreeArtifacts,
        expandedFilesets,
        pathResolver,
        importantInputMap,
        expandFilesets,
        fullyResolveFilesetSymlinks);
  }

  public ArtifactPathResolver pathResolver() {
    return pathResolver;
  }

  public ActionInputMap getImportantInputMap() {
    return importantInputMap;
  }

  public Map<Artifact, ImmutableList<FilesetOutputSymlink>> getExpandedFilesets() {
    return expandedFilesets;
  }

  @Nullable
  public FileArtifactValue getFileArtifactValue(Artifact artifact) {
    return importantInputMap.getInputMetadata(artifact);
  }

  /** Returns owner mappings for artifact expansions contained in {@code inputsOfInterest}. */
  public ActionInputDepOwners getDepOwners(Collection<ActionInput> inputsOfInterest) {
    ActionInputDepOwnerMap depOwners = new ActionInputDepOwnerMap(inputsOfInterest);
    expandedTreeArtifacts.forEach(
        (tree, children) -> {
          for (Artifact child : children) {
            depOwners.addOwner(child, tree);
          }
        });
    if (expandFilesets) {
      for (Artifact fileset : expandedFilesets.keySet()) {
        visitFileset(
            fileset,
            new ArtifactReceiver() {
              @Override
              public void accept(Artifact artifact) {
                throw new AssertionError(artifact);
              }

              @Override
              public void acceptFilesetMapping(
                  Artifact fileset, PathFragment relName, Path targetFile) {
                depOwners.addOwner(ActionInputHelper.fromPath(targetFile.asFragment()), fileset);
              }
            });
      }
    }
    return depOwners;
  }

  /** Visits the expansion of the given artifacts. */
  public void visitArtifacts(Iterable<Artifact> artifacts, ArtifactReceiver receiver) {
    for (Artifact artifact : artifacts) {
      if (artifact.isMiddlemanArtifact()) {
        continue;
      }
      if (artifact.isFileset()) {
        if (expandFilesets) {
          visitFileset(artifact, receiver);
        }
      } else if (artifact.isTreeArtifact()) {
        FileArtifactValue treeArtifactMetadata = importantInputMap.getInputMetadata(artifact);
        if (treeArtifactMetadata == null) {
          BugReport.sendBugReport(
              new IllegalStateException(
                  String.format(
                      "missing artifact metadata for tree artifact: %s",
                      artifact.toDebugString())));
        }
        if (FileArtifactValue.OMITTED_FILE_MARKER.equals(treeArtifactMetadata)) {
          // Expansion can be missing for omitted tree artifacts -- skip the whole tree.
          continue;
        }
        for (Artifact child :
            checkNotNull(
                expandedTreeArtifacts.get(artifact),
                "Missing expansion for tree artifact: %s",
                artifact)) {
          receiver.accept(child);
        }
      } else {
        receiver.accept(artifact);
      }
    }
  }

  private void visitFileset(Artifact filesetArtifact, ArtifactReceiver receiver) {
    ImmutableList<FilesetOutputSymlink> links = expandedFilesets.get(filesetArtifact);
    FilesetManifest filesetManifest =
        FilesetManifest.constructFilesetManifestWithoutError(
            links,
            PathFragment.EMPTY_FRAGMENT,
            fullyResolveFilesetLinks
                ? RelativeSymlinkBehaviorWithoutError.RESOLVE_FULLY
                : RelativeSymlinkBehaviorWithoutError.RESOLVE);

    for (Map.Entry<PathFragment, String> mapping : filesetManifest.getEntries().entrySet()) {
      String targetFile = mapping.getValue();
      PathFragment locationInFileset = mapping.getKey();
      receiver.acceptFilesetMapping(
          filesetArtifact, locationInFileset, execRoot.getRelative(targetFile));
    }
  }

  @Override
  public ImmutableSortedSet<TreeFileArtifact> expandTreeArtifact(Artifact treeArtifact) {
    checkArgument(treeArtifact.isTreeArtifact(), treeArtifact);
    return expandedTreeArtifacts.getOrDefault(treeArtifact, ImmutableSortedSet.of());
  }

  @Override
  public ImmutableList<FilesetOutputSymlink> expandFileset(Artifact fileset)
      throws MissingExpansionException {
    checkArgument(fileset.isFileset(), fileset);
    checkState(expandFilesets, "Fileset expansion disabled, cannot expand %s", fileset);
    ImmutableList<FilesetOutputSymlink> links = expandedFilesets.get(fileset);
    if (links == null) {
      throw new MissingExpansionException("Missing expansion for fileset: " + fileset);
    }
    return links;
  }

  /** A function that accepts an {@link Artifact}. */
  public interface ArtifactReceiver {
    void accept(Artifact artifact);

    void acceptFilesetMapping(Artifact fileset, PathFragment relName, Path targetFile);
  }

  /** A factory for {@link ArtifactPathResolver}. */
  public interface PathResolverFactory {
    ArtifactPathResolver createPathResolverForArtifactValues(
        ActionInputMap actionInputMap,
        Map<Artifact, ImmutableSortedSet<TreeFileArtifact>> treeArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets,
        String workspaceName);

    boolean shouldCreatePathResolverForArtifactValues();
  }
}
