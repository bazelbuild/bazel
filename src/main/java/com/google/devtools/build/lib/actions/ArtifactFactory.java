// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A cache of Artifacts, keyed by Path.
 */
@ThreadSafe
public class ArtifactFactory implements ArtifactResolver, ArtifactSerializer, ArtifactDeserializer {

  private final Path execRoot;

  /**
   * The main Path to Artifact cache. There will always be exactly one canonical
   * artifact for a given path.
   */
  private final Map<PathFragment, Artifact> pathToArtifact = new HashMap<>();

  /**
   * Map of package names to source root paths so that we can create source
   * artifact paths given execPaths in the symlink forest.
   */
  private ImmutableMap<PathFragment, Root> packageRoots;

  /**
   * Reverse-ordered list of derived roots for use in looking up or creating
   * derived artifacts from execPaths. The reverse order is only significant
   * for overlapping roots so that the longest is found first.
   */
  private ImmutableCollection<Root> derivedRoots = ImmutableList.of();

  /**
   * Whether to also keep track of derived artifacts in the {@link #pathToArtifact} map. If
   * set to false, a instance is returned upon each call that returns an {@link Artifact}.
   */
  private boolean reuseDerivedArtifacts;

  private ArtifactIdRegistry artifactIdRegistry = new ArtifactIdRegistry();

  /**
   * Constructs a new artifact factory that will use a given execution root when
   * creating artifacts.
   *
   * @param execRoot the execution root Path to use
   */
  public ArtifactFactory(Path execRoot) {
    this.execRoot = execRoot;
    this.reuseDerivedArtifacts = true;
  }

  /**
   * Clear the cache.
   */
  public synchronized void clear(boolean newReuseDerivedArtifacts) {
    pathToArtifact.clear();
    packageRoots = null;
    derivedRoots = ImmutableList.of();
    reuseDerivedArtifacts = newReuseDerivedArtifacts;
    artifactIdRegistry = new ArtifactIdRegistry();
  }

  /**
   * Set the set of known packages and their corresponding source artifact
   * roots. Must be called exactly once after construction or clear().
   *
   * @param packageRoots the map of package names to source artifact roots to
   *        use.
   */
  public synchronized void setPackageRoots(Map<PathFragment, Root> packageRoots) {
    this.packageRoots = ImmutableMap.copyOf(packageRoots);
  }

  /**
   * Set the set of known derived artifact roots. Must be called exactly once
   * after construction or clear().
   *
   * @param roots the set of derived artifact roots to use
   */
  public synchronized void setDerivedArtifactRoots(Collection<Root> roots) {
    derivedRoots = ImmutableSortedSet.<Root>reverseOrder().addAll(roots).build();
  }

  @Override
  public Artifact getSourceArtifact(PathFragment execPath, Root root, ArtifactOwner owner) {
    Preconditions.checkArgument(!execPath.isAbsolute());
    Preconditions.checkNotNull(owner, execPath);
    execPath = execPath.normalize();
    return getArtifact(root.getPath().getRelative(execPath), root, execPath, owner, null);
  }

  @Override
  public Artifact getSourceArtifact(PathFragment execPath, Root root) {
    return getSourceArtifact(execPath, root, ArtifactOwner.NULL_OWNER);
  }

  /**
   * Only for use by BinTools! Returns an artifact for a tool at the given path
   * fragment, relative to the exec root, creating it if not found. This method
   * only works for normalized, relative paths.
   */
  public Artifact getDerivedArtifact(PathFragment execPath) {
    Preconditions.checkArgument(!execPath.isAbsolute(), execPath);
    Preconditions.checkArgument(execPath.isNormalized(), execPath);
    // TODO(bazel-team): Check that either BinTools do not change over the life of the Blaze server,
    // or require that a legitimate ArtifactOwner be passed in here to allow for ownership.
    return getArtifact(execRoot.getRelative(execPath), Root.execRootAsDerivedRoot(execRoot),
        execPath, ArtifactOwner.NULL_OWNER, null);
  }

  private void validatePath(PathFragment rootRelativePath, Root root) {
    Preconditions.checkArgument(!rootRelativePath.isAbsolute(), rootRelativePath);
    Preconditions.checkArgument(rootRelativePath.isNormalized(), rootRelativePath);
    Preconditions.checkArgument(root.getPath().startsWith(execRoot), "%s %s", root, execRoot);
    Preconditions.checkArgument(!root.getPath().equals(execRoot), "%s %s", root, execRoot);
    // TODO(bazel-team): this should only accept roots from derivedRoots.
    //Preconditions.checkArgument(derivedRoots.contains(root), "%s not in %s", root, derivedRoots);
  }

  /**
   * Returns an artifact for a tool at the given root-relative path under the given root, creating
   * it if not found. This method only works for normalized, relative paths.
   *
   * <p>The root must be below the execRoot, and the execPath of the resulting Artifact is computed
   * as {@code root.getRelative(rootRelativePath).relativeTo(execRoot)}.
   */
  // TODO(bazel-team): Don't allow root == execRoot.
  public Artifact getDerivedArtifact(PathFragment rootRelativePath, Root root,
      ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    Path path = root.getPath().getRelative(rootRelativePath);
    return getArtifact(path, root, path.relativeTo(execRoot), owner, null);
  }

  /**
   * Returns an artifact that represents the output directory of a Fileset at the given
   * root-relative path under the given root, creating it if not found. This method only works for
   * normalized, relative paths.
   *
   * <p>The root must be below the execRoot, and the execPath of the resulting Artifact is computed
   * as {@code root.getRelative(rootRelativePath).relativeTo(execRoot)}.
   */
  public Artifact getFilesetArtifact(PathFragment rootRelativePath, Root root,
      ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    Path path = root.getPath().getRelative(rootRelativePath);
    return getArtifact(path, root, path.relativeTo(execRoot), owner, SpecialArtifactType.FILESET);
  }

  public Artifact getSpecialMetadataHandlingArtifact(PathFragment rootRelativePath, Root root,
      ArtifactOwner owner, boolean forceConstantMetadata, boolean forceDigestMetadata) {
    validatePath(rootRelativePath, root);
    Path path = root.getPath().getRelative(rootRelativePath);
    SpecialArtifactType type = null;
    if (forceConstantMetadata) {
      Preconditions.checkArgument(!forceDigestMetadata);
      type = SpecialArtifactType.FORCE_CONSTANT_METADATA;
    } else if (forceDigestMetadata) {
      type = SpecialArtifactType.FORCE_DIGEST_METADATA;
    }
    return getArtifact(path, root, path.relativeTo(execRoot), owner, type);
  }

  /**
   * Returns the artifact at the specified location under the specified root if it exists. If it
   * does not, returns null, unless noReuseDerivedArtifacts is set, in which case it will always
   * return an artifact.
   */
  public synchronized Artifact getExistingDerivedArtifact(
      PathFragment rootRelativePath, Root root, ArtifactOwner owner) {
    if (!reuseDerivedArtifacts) {
      return getDerivedArtifact(rootRelativePath, root, owner);
    }

    Preconditions.checkState(!root.isSourceRoot());
    validatePath(rootRelativePath, root);
    Path path = root.getPath().getRelative(rootRelativePath);
    PathFragment execPath = path.relativeTo(execRoot);
    Artifact result = pathToArtifact.get(execPath);
    Preconditions.checkState(result == null || owner == ArtifactOwner.NULL_OWNER
        || result.getArtifactOwner().equals(owner), "%s has owner %s but %s was specified", result,
        result == null ? null : result.getArtifactOwner(), owner);
    return result;
  }

  /**
   * Returns the fileset artifact at the specified location under the specified root if it exists.
   * If it does not, returns null, unless noReuseDerivedArtifacts is set, in which case it will
   * always return an artifact.
   */
  public synchronized Artifact getExistingFilesetArtifact(
      PathFragment rootRelativePath, Root root, ArtifactOwner owner) {
    if (!reuseDerivedArtifacts) {
      return getFilesetArtifact(rootRelativePath, root, owner);
    }

    Preconditions.checkState(!root.isSourceRoot());
    validatePath(rootRelativePath, root);
    Path path = root.getPath().getRelative(rootRelativePath);
    PathFragment execPath = path.relativeTo(execRoot);
    Artifact result = pathToArtifact.get(execPath);
    Preconditions.checkState(result == null || result.getArtifactOwner().equals(owner),
        "%s has owner %s but %s was specified", result,
        result == null ? null : result.getArtifactOwner(), owner);
    return result;
  }

  /**
   * Returns the Artifact for the specified path, creating one if not found and
   * setting the <code>root</code> and <code>execPath</code> to the
   * specified values.
   */
  private synchronized Artifact getArtifact(Path path, Root root, PathFragment execPath,
      ArtifactOwner owner, @Nullable SpecialArtifactType type) {
    Preconditions.checkNotNull(root);
    Preconditions.checkNotNull(execPath);

    if (!reuseDerivedArtifacts && !root.isSourceRoot()) {
      return createArtifact(path, root, execPath, owner, type);
    }

    Artifact artifact = pathToArtifact.get(execPath);

    if (artifact == null || !Objects.equals(artifact.getArtifactOwner(), owner)) {
      // There really should be a safety net that makes it impossible to create two Artifacts
      // with the same exec path but a different Owner, but we also need to reuse Artifacts from
      // previous builds.
      artifact = createArtifact(path, root, execPath, owner, type);
      pathToArtifact.put(execPath, artifact);
    } else {
      // TODO(bazel-team): Maybe we should check for equality of the fileset bit. However, that
      // would require us to differentiate between artifact-creating and artifact-getting calls to
      // getDerivedArtifact().
      Preconditions.checkState(root.equals(artifact.getRoot()),
          "root for path %s changed from %s to %s", path, artifact.getRoot(), root);
      Preconditions.checkState(execPath.equals(artifact.getExecPath()),
          "execPath for path %s changed from %s to %s", path, artifact.getExecPath(), execPath);
    }
    return artifact;
  }

  private Artifact createArtifact(Path path, Root root, PathFragment execPath, ArtifactOwner owner,
      @Nullable SpecialArtifactType type) {
    Preconditions.checkNotNull(owner, path);
    if (type == null) {
      return new Artifact(path, root, execPath, owner);
    } else {
      return new Artifact.SpecialArtifact(path, root, execPath, owner, type);
    }
  }

  /**
   * Returns true if the artifact factory still contains an artifact
   * corresponding to the exec path of the given artifact.
   */
  public synchronized boolean artifactExists(Artifact artifact) {
    if (!reuseDerivedArtifacts) {
      return true;  // we are amnesic; assume that it does.
    }
    return pathToArtifact.containsKey(artifact.getExecPath());
  }

  /**
   * Removes given Artifact from the artifact factory.
   *
   * <p> Note: It is a responsibility of the caller to be sure that the artifact is not referenced
   * by any action on the dependency graph. {@link com.google.devtools.build.lib.view.BuildView}
   * class will validate that all artifacts on the dependency graph are present in the artifact
   * factory.
   */
  public synchronized void removeArtifact(Artifact artifact) {
    Preconditions.checkNotNull(artifact);
    pathToArtifact.remove(artifact.getExecPath());
  }

  /**
   * Removes the scheduling middleman from the internal artifact storage. The caller of this
   * function must ensure that there are no actions that use the middleman that is being removed.
   *
   * @param middleman the scheduling middleman to remove
   */
  public synchronized void removeSchedulingMiddleman(Artifact middleman) {
    removeArtifact(middleman);
  }

  @Override
  public synchronized Artifact resolveSourceArtifact(PathFragment execPath) {
    Artifact result = internalResolveArtifact(execPath, false, ArtifactOwner.NULL_OWNER);
    return result != null && result.isSourceArtifact() ? result : null;
  }

  private Artifact internalResolveArtifact(PathFragment execPath, boolean createDerivedArtifacts,
      ArtifactOwner owner) {
    execPath = execPath.normalize();
    // First try a quick map lookup to see if the artifact already exists.
    Artifact a = pathToArtifact.get(execPath);
    if (a != null) {
      return a;
    }
    // See if the path starts with one of the derived roots, & create a derived Artifact if so.
    Path path = execRoot.getRelative(execPath);
    Root derivedRoot = findDerivedRoot(path);
    if (derivedRoot != null) {
      if (createDerivedArtifacts) {
        return getDerivedArtifact(path.relativeTo(derivedRoot.getPath()), derivedRoot, owner);
      } else {
        return null;
      }
    }
    // Must be a new source artifact, so probe the known packages to find the longest package
    // prefix, and then use the corresponding source root to create a new artifact.
    for (PathFragment dir = execPath.getParentDirectory(); dir != null;
         dir = dir.getParentDirectory()) {
      Root sourceRoot = packageRoots.get(dir);
      if (sourceRoot != null) {
        return getSourceArtifact(execPath, sourceRoot, owner);
      }
    }
    return null;  // not a path that we can find...
  }

  /**
   * Finds the derived root for a full path by comparing against the known
   * derived artifact roots.
   *
   * @param path a Path to resolve the root for
   * @return the root for the path or null if no root can be determined
   */
  @VisibleForTesting  // for our own unit tests only.
  synchronized Root findDerivedRoot(Path path) {
    for (Root prefix : derivedRoots) {
      if (path.startsWith(prefix.getPath())) {
        return prefix;
      }
    }
    return null;
  }

  /**
   * Returns all artifacts created by the artifact factory.
   */
  public synchronized Collection<Artifact> getArtifacts() {
    return ImmutableList.copyOf(pathToArtifact.values());
  }


  /**
   * Resolves an artifact based on its deserialized representation. The artifact can be either a
   * source or a derived one.
   *
   * <p>Note: this method represents a hole in the usual contract that artifacts with a random path
   * cannot be created. Unfortunately, we currently need this in some cases.
   *
   * @param execPath the exec path of the artifact
   * @param isFileset whether the artifact is the output of a fileset
   * @param owner the owner of the artifact.
   */
  // TODO(bazel-team): This probably doesn't work. We may need to know the actual configured
  // target that is deserializing this artifact, not just the label. [skyframe-execution]
  public Artifact deserializeArtifact(PathFragment execPath, boolean isFileset, Label owner) {
    ArtifactOwner artifactOwner = new LabelArtifactOwner(owner);
    if (reuseDerivedArtifacts) {
      return internalResolveArtifact(execPath, true, artifactOwner);
    }

    Path path = execRoot.getRelative(execPath);
    Root root = findDerivedRoot(path);

    Artifact result;
    if (root != null) {
      Preconditions.checkState(owner == null);
      result = isFileset
          ? getFilesetArtifact(path.relativeTo(root.getPath()), root, artifactOwner)
          : getDerivedArtifact(path.relativeTo(root.getPath()), root, artifactOwner);
    } else {
      for (PathFragment dir = execPath.getParentDirectory(); dir != null;
          dir = dir.getParentDirectory()) {
        root = packageRoots.get(dir);
        if (root != null) {
          break;
        }
      }

      if (root == null) {
        // Root not found. Return null to indicate that we could not create the artifact.
        return null;
      }

      result = getSourceArtifact(execPath, root, artifactOwner);
    }

    return result;
  }

  @Override
  public Artifact lookupArtifactById(int artifactId) {
    return artifactIdRegistry.lookupArtifactById(artifactId);
  }

  @Override
  public ImmutableList<Artifact> lookupArtifactsByIds(Iterable<Integer> artifactIds) {
    return artifactIdRegistry.lookupArtifactsByIds(artifactIds);
  }

  @Override
  public int getArtifactId(Artifact artifact) {
    return artifactIdRegistry.getArtifactId(artifact);
  }

  /**
   * ArtifactOwner wrapper for Labels.
   */
  @VisibleForTesting
  public static class LabelArtifactOwner implements ArtifactOwner {
    private final Label label;

    @VisibleForTesting
    public LabelArtifactOwner(Label label) {
      this.label = label;
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    public int hashCode() {
      return label == null ? super.hashCode() : label.hashCode();
    }

    @Override
    public boolean equals(Object that) {
      if (!(that instanceof LabelArtifactOwner)) {
        return false;
      }
      return Objects.equals(this.label, ((LabelArtifactOwner) that).label);
    }
  }
}
