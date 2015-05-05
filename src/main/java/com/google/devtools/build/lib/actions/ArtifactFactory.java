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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import javax.annotation.Nullable;

/**
 * A cache of Artifacts, keyed by Path.
 */
@ThreadSafe
public class ArtifactFactory implements ArtifactResolver, ArtifactSerializer, ArtifactDeserializer {

  private final Path execRoot;

  /**
   * The main Path to source artifact cache. There will always be exactly one canonical
   * artifact for a given source path.
   */
  private final Map<PathFragment, Artifact> pathToSourceArtifact = new HashMap<>();

  /**
   * Map of package names to source root paths so that we can create source
   * artifact paths given execPaths in the symlink forest.
   */
  private ImmutableMap<PackageIdentifier, Root> packageRoots;

  /**
   * Reverse-ordered list of derived roots for use in looking up or (in rare cases) creating
   * derived artifacts from execPaths. The reverse order is only significant for overlapping roots
   * so that the longest is found first.
   */
  private ImmutableCollection<Root> derivedRoots = ImmutableList.of();

  private ArtifactIdRegistry artifactIdRegistry = new ArtifactIdRegistry();

  /**
   * Constructs a new artifact factory that will use a given execution root when
   * creating artifacts.
   *
   * @param execRoot the execution root Path to use
   */
  public ArtifactFactory(Path execRoot) {
    this.execRoot = execRoot;
  }

  /**
   * Clear the cache.
   */
  public synchronized void clear() {
    pathToSourceArtifact.clear();
    packageRoots = null;
    derivedRoots = ImmutableList.of();
    artifactIdRegistry = new ArtifactIdRegistry();
    clearDeserializedArtifacts();
  }

  /**
   * Set the set of known packages and their corresponding source artifact
   * roots. Must be called exactly once after construction or clear().
   *
   * @param packageRoots the map of package names to source artifact roots to
   *        use.
   */
  public synchronized void setPackageRoots(Map<PackageIdentifier, Root> packageRoots) {
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

  public Artifact getConstantMetadataArtifact(PathFragment rootRelativePath, Root root,
      ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    Path path = root.getPath().getRelative(rootRelativePath);
    return getArtifact(
        path, root, path.relativeTo(execRoot), owner, SpecialArtifactType.CONSTANT_METADATA);
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

    if (!root.isSourceRoot()) {
      return createArtifact(path, root, execPath, owner, type);
    }

    Artifact artifact = pathToSourceArtifact.get(execPath);

    if (artifact == null || !Objects.equals(artifact.getArtifactOwner(), owner)) {
      // There really should be a safety net that makes it impossible to create two Artifacts
      // with the same exec path but a different Owner, but we also need to reuse Artifacts from
      // previous builds.
      artifact = createArtifact(path, root, execPath, owner, type);
      pathToSourceArtifact.put(execPath, artifact);
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

  @Override
  public synchronized Artifact resolveSourceArtifact(PathFragment execPath) {
    execPath = execPath.normalize();
    if (execPath.containsUplevelReferences()) {
      // Source exec paths cannot escape the source root.
      return null;
    }
    // First try a quick map lookup to see if the artifact already exists.
    Artifact a = pathToSourceArtifact.get(execPath);
    if (a != null) {
      return a;
    }
    // Don't create an artifact if it's derived.
    if (findDerivedRoot(execRoot.getRelative(execPath)) != null) {
      return null;
    }
    // Must be a new source artifact, so probe the known packages to find the longest package
    // prefix, and then use the corresponding source root to create a new artifact.
    for (PathFragment dir = execPath.getParentDirectory(); dir != null;
         dir = dir.getParentDirectory()) {
      Root sourceRoot = packageRoots.get(PackageIdentifier.createInDefaultRepo(dir));
      if (sourceRoot != null) {
        return getSourceArtifact(execPath, sourceRoot, ArtifactOwner.NULL_OWNER);
      }
    }
    return null;  // not a path that we can find...
  }

  @Override
  public synchronized Map<PathFragment, Artifact> resolveSourceArtifacts(
      Iterable<PathFragment> execPaths, PackageRootResolver resolver) {
    Map<PathFragment, Artifact> result = new HashMap<>();
    ArrayList<PathFragment> unresolvedPaths = new ArrayList<>();

    for (PathFragment execPath : execPaths) {
      PathFragment execPathNormalized = execPath.normalize();
      if (execPathNormalized.containsUplevelReferences()) {
        // Source exec paths cannot escape the source root.
        result.put(execPath, null);
        continue;
      }
      // First try a quick map lookup to see if the artifact already exists.
      Artifact a = pathToSourceArtifact.get(execPathNormalized);
      if (a != null) {
        result.put(execPath, a);
      } else if (findDerivedRoot(execRoot.getRelative(execPathNormalized)) != null) {
        // Don't create an artifact if it's derived.
        result.put(execPath, null);
      } else {
        // Remember this path, maybe we can resolve it with the help of PackageRootResolver.
        unresolvedPaths.add(execPath);
      }
    }
    Map<PathFragment, Root> sourceRoots = resolver.findPackageRoots(unresolvedPaths);
    // We are missing some dependencies. We need to rerun this method later.
    if (sourceRoots == null) {
      return null;
    }
    for (PathFragment path : unresolvedPaths) {
      Root sourceRoot = sourceRoots.get(path);
      if (sourceRoot != null) {
        // We have found corresponding source root, so we should create a new source artifact.
        result.put(path, getSourceArtifact(path.normalize(), sourceRoot, ArtifactOwner.NULL_OWNER));
      } else {
        // Not a path that we can find...
        result.put(path, null);
      }
    }
    return result;
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
   * Returns all source artifacts created by the artifact factory.
   */
  public synchronized Iterable<Artifact> getSourceArtifacts() {
    return ImmutableList.copyOf(pathToSourceArtifact.values());
  }

  // Non-final only because clear()ing a map does not actually free the memory it took up, so we
  // assign it to a new map in lieu of clearing.
  private ConcurrentMap<PathFragment, Artifact> deserializedArtifacts =
      new ConcurrentHashMap<>();

  /**
   * Returns the map of all artifacts that were deserialized this build. The caller should process
   * them and then call {@link #clearDeserializedArtifacts}.
   */
  public Map<PathFragment, Artifact> getDeserializedArtifacts() {
    return deserializedArtifacts;
  }

  /** Clears the map of deserialized artifacts. */
  public void clearDeserializedArtifacts() {
    deserializedArtifacts = new ConcurrentHashMap<>();
  }

  /**
   * Resolves an artifact based on its deserialized representation. The artifact can be either a
   * source or a derived one.
   *
   * <p>Note: this method represents a hole in the usual contract that artifacts with a random path
   * cannot be created. Unfortunately, we currently need this in some cases.
   *
   * @param execPath the exec path of the artifact
   */
  public Artifact deserializeArtifact(PathFragment execPath, PackageRootResolver resolver) {
    Preconditions.checkArgument(!execPath.isAbsolute(), execPath);
    Path path = execRoot.getRelative(execPath);
    Root root = findDerivedRoot(path);

    Artifact result;
    if (root != null) {
      result = getDerivedArtifact(path.relativeTo(root.getPath()), root,
          Artifact.DESERIALIZED_MARKER_OWNER);
      Artifact oldResult = deserializedArtifacts.putIfAbsent(execPath, result);
      if (oldResult != null) {
        result = oldResult;
      }
      return result;
    } else {
      Map<PathFragment, Root> sourceRoots = resolver.findPackageRoots(Lists.newArrayList(execPath));
      if (sourceRoots == null || sourceRoots.get(execPath) == null) {
        return null;
      }
      return getSourceArtifact(execPath, sourceRoots.get(execPath), ArtifactOwner.NULL_OWNER);
    }
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
}
