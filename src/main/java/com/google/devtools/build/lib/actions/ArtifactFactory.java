// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Striped;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import javax.annotation.Nullable;

/** A cache of Artifacts, keyed by Path. */
@ThreadSafe
public class ArtifactFactory implements ArtifactResolver {
  private static final int CONCURRENCY_LEVEL = Runtime.getRuntime().availableProcessors();
  private static final Striped<Lock> STRIPED_LOCK = Striped.lock(CONCURRENCY_LEVEL);

  private final Path execRoot;
  private final Path execRootParent;
  private final PathFragment derivedPathPrefix;
  private ImmutableMap<Root, ArtifactRoot> sourceArtifactRoots;

  /**
   * Cache of source artifacts.
   */
  private final SourceArtifactCache sourceArtifactCache = new SourceArtifactCache();

  /**
   * Map of package names to source root paths so that we can create source artifact paths given
   * execPaths in the symlink forest.
   */
  private PackageRoots.PackageRootLookup packageRoots;

  private static class SourceArtifactCache {

    private class Entry {
      private final Artifact artifact;
      private final int idOfBuild;

      Entry(Artifact artifact) {
        this.artifact = artifact;
        idOfBuild = buildId;
      }

      Artifact getArtifact() {
        return artifact;
      }

      int getIdOfBuild() {
        return idOfBuild;
      }
    }

    /**
     * The main Path to source artifact cache. There will always be exactly one canonical artifact
     * for a given source path.
     */
    private final Map<PathFragment, Entry> pathToSourceArtifact = new ConcurrentHashMap<>();

    /** Id of current build. Has to be increased every time before execution phase starts. */
    private int buildId = 0;

    /** Returns artifact if it present in the cache, otherwise null. */
    @ThreadSafe
    Artifact getArtifact(PathFragment execPath) {
      Entry cacheEntry = pathToSourceArtifact.get(execPath);
      return cacheEntry == null ? null : cacheEntry.getArtifact();
    }

    /**
     * Returns artifact if it present in the cache and was created during this build, otherwise
     * null.
     */
    @ThreadSafe
    Artifact getArtifactIfValid(PathFragment execPath) {
      Entry cacheEntry = pathToSourceArtifact.get(execPath);
      if (cacheEntry != null && cacheEntry.getIdOfBuild() == buildId) {
        return cacheEntry.getArtifact();
      }
      return null;
    }

    @ThreadCompatible // Calls #putArtifact.
    void markEntryAsValid(PathFragment execPath) {
      Artifact oldValue = Preconditions.checkNotNull(getArtifact(execPath));
      putArtifact(execPath, oldValue);
    }

    void newBuild() {
      buildId++;
    }

    void clear() {
      pathToSourceArtifact.clear();
      buildId = 0;
    }

    @ThreadCompatible // Concurrent puts do not know which one actually got its artifact in.
    void putArtifact(PathFragment execPath, Artifact artifact) {
      pathToSourceArtifact.put(execPath, new Entry(artifact));
    }
  }

  /**
   * Constructs a new artifact factory that will use a given execution root when creating artifacts.
   *
   * @param execRoot the execution root Path to use. This will be
   *     [output_base]/execroot/[workspace].
   */
  public ArtifactFactory(Path execRoot, String derivedPathPrefix) {
    this.execRoot = execRoot;
    this.execRootParent = execRoot.getParentDirectory();
    this.derivedPathPrefix = PathFragment.create(derivedPathPrefix);
  }

  /**
   * Clear the cache.
   */
  public synchronized void clear() {
    packageRoots = null;
    sourceArtifactCache.clear();
  }

  public synchronized void setSourceArtifactRoots(
      ImmutableMap<Root, ArtifactRoot> sourceArtifactRoots) {
    this.sourceArtifactRoots = sourceArtifactRoots;
  }

  /**
   * Set the set of known packages and their corresponding source artifact roots. Must be called
   * exactly once after construction or clear().
   *
   * @param packageRoots provider of a source root given a package identifier.
   */
  public synchronized void setPackageRoots(PackageRoots.PackageRootLookup packageRoots) {
    this.packageRoots = packageRoots;
    sourceArtifactCache.newBuild();
  }

  @Override
  public Artifact getSourceArtifact(PathFragment execPath, Root root, ArtifactOwner owner) {
    Preconditions.checkArgument(
        execPath.isAbsolute() == root.isAbsolute(), "%s %s %s", execPath, root, owner);
    Preconditions.checkNotNull(owner, "%s %s", execPath, root);
    Preconditions.checkNotNull(
        sourceArtifactRoots, "Not initialized for %s %s %s", execPath, root, owner);
    return getArtifact(
        Preconditions.checkNotNull(
            sourceArtifactRoots.get(root), "%s has no ArtifactRoot (%s)", root, execPath),
        execPath,
        owner,
        null);
  }

  @Override
  public Artifact getSourceArtifact(PathFragment execPath, Root root) {
    return getSourceArtifact(execPath, root, ArtifactOwner.NullArtifactOwner.INSTANCE);
  }

  private void validatePath(PathFragment rootRelativePath, ArtifactRoot root) {
    Preconditions.checkArgument(!root.isSourceRoot());
    Preconditions.checkArgument(
        rootRelativePath.isAbsolute() == root.getRoot().isAbsolute(), rootRelativePath);
    Preconditions.checkArgument(!rootRelativePath.containsUplevelReferences(), rootRelativePath);
    Preconditions.checkArgument(
        root.getRoot().asPath().startsWith(execRootParent),
        "%s must start with %s, root = %s, root fs = %s, execRootParent fs = %s",
        root.getRoot(),
        execRootParent,
        root,
        root.getRoot().asPath().getFileSystem(),
        execRootParent.getFileSystem());
    Preconditions.checkArgument(
        !root.getRoot().asPath().equals(execRootParent),
        "%s %s %s",
        root.getRoot(),
        execRootParent,
        root);
    // TODO(bazel-team): this should only accept roots from derivedRoots.
    //Preconditions.checkArgument(derivedRoots.contains(root), "%s not in %s", root, derivedRoots);
  }

  /**
   * Returns an artifact for a tool at the given root-relative path under the given root, creating
   * it if not found. This method only works for normalized, relative paths.
   *
   * <p>The root must be below the execRootParent, and the execPath of the resulting Artifact is
   * computed as {@code root.getRelative(rootRelativePath).relativeTo(root.execRoot)}.
   */
  // TODO(bazel-team): Don't allow root == execRootParent.
  public Artifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return getArtifact(root, root.getExecPath().getRelative(rootRelativePath), owner, null);
  }

  /**
   * Returns an artifact that represents the output directory of a Fileset at the given
   * root-relative path under the given root, creating it if not found. This method only works for
   * normalized, relative paths.
   *
   * <p>The root must be below the execRootParent, and the execPath of the resulting Artifact is
   * computed as {@code root.getRelative(rootRelativePath).relativeTo(root.execRoot)}.
   */
  public Artifact getFilesetArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return getArtifact(
        root, root.getExecPath().getRelative(rootRelativePath), owner, SpecialArtifactType.FILESET);
  }

  /**
   * Returns an artifact that represents a TreeArtifact; that is, a directory containing some tree
   * of ArtifactFiles unknown at analysis time.
   *
   * <p>The root must be below the execRootParent, and the execPath of the resulting Artifact is
   * computed as {@code root.getRelative(rootRelativePath).relativeTo(root.execRoot)}.
   */
  public Artifact getTreeArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return getArtifact(
        root, root.getExecPath().getRelative(rootRelativePath), owner, SpecialArtifactType.TREE);
  }

  public Artifact getConstantMetadataArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return getArtifact(
        root,
        root.getExecPath().getRelative(rootRelativePath),
        owner,
        SpecialArtifactType.CONSTANT_METADATA);
  }

  /**
   * Returns the Artifact for the specified path, creating one if not found and setting the <code>
   * root</code> and <code>execPath</code> to the specified values.
   */
  private Artifact getArtifact(
      ArtifactRoot root,
      PathFragment execPath,
      ArtifactOwner owner,
      @Nullable SpecialArtifactType type) {
    Preconditions.checkNotNull(root);
    Preconditions.checkNotNull(execPath);

    if (!root.isSourceRoot()) {
      return createArtifact(root, execPath, owner, type);
    }

    // Double-checked locking to avoid locking cost when possible.
    Artifact artifact = sourceArtifactCache.getArtifact(execPath);
    if (artifact == null || !Objects.equals(artifact.getArtifactOwner(), owner)
        || !root.equals(artifact.getRoot())) {
      Lock lock = STRIPED_LOCK.get(execPath);
      lock.lock();
      try {
        artifact = sourceArtifactCache.getArtifact(execPath);
        if (artifact == null
            || !Objects.equals(artifact.getArtifactOwner(), owner)
            || !root.equals(artifact.getRoot())) {
          // There really should be a safety net that makes it impossible to create two Artifacts
          // with the same exec path but a different Owner, but we also need to reuse Artifacts from
          // previous builds.
          artifact = createArtifact(root, execPath, owner, type);
          sourceArtifactCache.putArtifact(execPath, artifact);
        }
      } finally {
        lock.unlock();
      }
    }
    return artifact;
  }

  private Artifact createArtifact(
      ArtifactRoot root,
      PathFragment execPath,
      ArtifactOwner owner,
      @Nullable SpecialArtifactType type) {
    Preconditions.checkNotNull(owner);
    if (type == null) {
      return root.isSourceRoot()
          ? new Artifact.SourceArtifact(root, execPath, owner)
          : new Artifact(root, execPath, owner);
    } else {
      return new Artifact.SpecialArtifact(root, execPath, owner, type);
    }
  }

  /**
   * Returns an {@link Artifact} with exec path formed by composing {@code baseExecPath} and {@code
   * relativePath} (via {@code baseExecPath.getRelative(relativePath)} if baseExecPath is not null).
   * That Artifact will have root determined by the package roots of this factory if it lives in a
   * subpackage distinct from that of baseExecPath, and {@code baseRoot} otherwise.
   */
  public synchronized Artifact resolveSourceArtifactWithAncestor(
      PathFragment relativePath,
      PathFragment baseExecPath,
      ArtifactRoot baseRoot,
      RepositoryName repositoryName) {
    Preconditions.checkState(
        (baseExecPath == null) == (baseRoot == null),
        "%s %s %s",
        relativePath,
        baseExecPath,
        baseRoot);
    Preconditions.checkState(
        !relativePath.isEmpty(), "%s %s %s", relativePath, baseExecPath, baseRoot);
    PathFragment execPath =
        baseExecPath != null ? baseExecPath.getRelative(relativePath) : relativePath;
    if (execPath.containsUplevelReferences()) {
      // Source exec paths cannot escape the source root.
      return null;
    }
    // Don't create an artifact if it's derived.
    if (isDerivedArtifact(execPath)) {
      return null;
    }
    Root sourceRoot =
        findSourceRoot(
            execPath, baseExecPath, baseRoot == null ? null : baseRoot.getRoot(), repositoryName);
    Artifact artifact = sourceArtifactCache.getArtifactIfValid(execPath);
    if (artifact != null) {
      ArtifactRoot artifactRoot = artifact.getRoot();
      Preconditions.checkState(
          sourceRoot == null || sourceRoot.equals(artifactRoot.getRoot()),
          "roots mismatch: %s %s %s",
          sourceRoot,
          artifactRoot,
          artifact);
      return artifact;
    }
    return createArtifactIfNotValid(sourceRoot, execPath);
  }

  /**
   * Probe the known packages to find the longest package prefix up until the base, or until the
   * root directory if our execPath doesn't start with baseExecPath due to uplevel references.
   */
  @Nullable
  private Root findSourceRoot(
      PathFragment execPath,
      @Nullable PathFragment baseExecPath,
      @Nullable Root baseRoot,
      RepositoryName repositoryName) {
    PathFragment dir = execPath.getParentDirectory();
    if (dir == null) {
      return null;
    }

    Pair<RepositoryName, PathFragment> repo = RepositoryName.fromPathFragment(dir);
    if (repo != null) {
      repositoryName = repo.getFirst();
      dir = repo.getSecond();
    }

    while (dir != null && !dir.equals(baseExecPath)) {
      Root sourceRoot =
          packageRoots.getRootForPackage(PackageIdentifier.create(repositoryName, dir));
      if (sourceRoot != null) {
        return sourceRoot;
      }
      dir = dir.getParentDirectory();
    }

    return dir != null && dir.equals(baseExecPath) ? baseRoot : null;
  }

  @Override
  public Artifact resolveSourceArtifact(PathFragment execPath,
      @SuppressWarnings("unused") RepositoryName repositoryName) {
    return resolveSourceArtifactWithAncestor(execPath, null, null, repositoryName);
  }

  @Override
  public synchronized Map<PathFragment, Artifact> resolveSourceArtifacts(
      Iterable<PathFragment> execPaths, PackageRootResolver resolver) throws InterruptedException {
    Map<PathFragment, Artifact> result = new HashMap<>();
    ArrayList<PathFragment> unresolvedPaths = new ArrayList<>();

    for (PathFragment execPath : execPaths) {
      if (execPath.containsUplevelReferences()) {
        // Source exec paths cannot escape the source root.
        result.put(execPath, null);
        continue;
      }
      // First try a quick map lookup to see if the artifact already exists.
      Artifact a = sourceArtifactCache.getArtifactIfValid(execPath);
      if (a != null) {
        result.put(execPath, a);
      } else if (isDerivedArtifact(execPath)) {
        // Don't create an artifact if it's derived.
        result.put(execPath, null);
      } else {
        // Remember this path, maybe we can resolve it with the help of PackageRootResolver.
        unresolvedPaths.add(execPath);
      }
    }
    Map<PathFragment, Root> sourceRoots = resolver.findPackageRootsForFiles(unresolvedPaths);
    // We are missing some dependencies. We need to rerun this method later.
    if (sourceRoots == null) {
      return null;
    }
    for (PathFragment path : unresolvedPaths) {
      result.put(path, createArtifactIfNotValid(sourceRoots.get(path), path));
    }
    return result;
  }

  @Override
  public Path getPathFromSourceExecPath(PathFragment execPath) {
    Preconditions.checkState(
        !execPath.startsWith(derivedPathPrefix), "%s is derived: %s", execPath, derivedPathPrefix);
    Root sourceRoot =
        packageRoots.getRootForPackage(PackageIdentifier.create(RepositoryName.MAIN, execPath));
    if (sourceRoot != null) {
      return sourceRoot.getRelative(execPath);
    }
    return execRoot.getRelative(execPath);
  }

  private Artifact createArtifactIfNotValid(Root sourceRoot, PathFragment execPath) {
    if (sourceRoot == null) {
      return null;  // not a path that we can find...
    }
    Artifact artifact = sourceArtifactCache.getArtifact(execPath);
    if (artifact != null && sourceRoot.equals(artifact.getRoot().getRoot())) {
      // Source root of existing artifact hasn't changed so we should mark corresponding entry in
      // the cache as valid.
      sourceArtifactCache.markEntryAsValid(execPath);
    } else {
      // Must be a new artifact or artifact in the cache is stale, so create a new one.
      artifact = getSourceArtifact(execPath, sourceRoot, ArtifactOwner.NullArtifactOwner.INSTANCE);
    }
    return artifact;
  }

  /**
   * Determines if an artifact is derived, that is, its root is a derived root or its exec path
   * starts with the bazel-out prefix.
   *
   * @param execPath The artifact's exec path.
   */
  @VisibleForTesting  // for our own unit tests only.
  synchronized boolean isDerivedArtifact(PathFragment execPath) {
    return execPath.startsWith(derivedPathPrefix);
  }
}
