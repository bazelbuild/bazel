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
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/** A cache of Artifacts, keyed by Path. */
@ThreadSafe
public class ArtifactFactory implements ArtifactResolver {

  private final Path execRootParent;
  private final PathFragment derivedPathPrefix;
  private ImmutableMap<Root, ArtifactRoot> sourceArtifactRoots;
  private boolean siblingRepositoryLayout = false;

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
      private final SourceArtifact artifact;
      private final int idOfBuild;

      Entry(SourceArtifact artifact) {
        this.artifact = artifact;
        idOfBuild = buildId;
      }

      SourceArtifact getArtifact() {
        return artifact;
      }

      boolean isArtifactValid() {
        return idOfBuild == buildId;
      }
    }

    private static final int CONCURRENCY_LEVEL = Runtime.getRuntime().availableProcessors();

    /**
     * The main Path to source artifact cache. There will always be exactly one canonical artifact
     * for a given source path.
     */
    private final ConcurrentMap<PathFragment, Entry> pathToSourceArtifact =
        new ConcurrentHashMap<>(16, 0.75f, CONCURRENCY_LEVEL);

    /** Id of current build. Has to be increased every time before analysis starts. */
    private int buildId = -1;

    /** Returns artifact if it present in the cache, otherwise null. */
    @ThreadSafe
    SourceArtifact getArtifact(PathFragment execPath) {
      Entry cacheEntry = pathToSourceArtifact.get(execPath);
      return cacheEntry == null ? null : cacheEntry.getArtifact();
    }

    /**
     * Returns artifact if it is present in the cache and has been verified to be valid for this
     * build, otherwise null. Note that if the artifact's package is not part of the current build,
     * our differing methods of validating source roots (via {@link PackageRootResolver} and via
     * {@link #findSourceRoot}) may disagree. In that case, the artifact will be valid, but unusable
     * by any action (since no action has properly declared it as an input).
     */
    @ThreadSafe
    Artifact getArtifactIfValid(PathFragment execPath) {
      Entry cacheEntry = pathToSourceArtifact.get(execPath);
      return (cacheEntry == null || !cacheEntry.isArtifactValid())
          ? null
          : cacheEntry.getArtifact();
    }

    void newBuild() {
      buildId++;
    }

    void clear() {
      pathToSourceArtifact.clear();
      buildId = -1;
    }
  }

  /**
   * Constructs a new artifact factory that will use a given execution root when creating artifacts.
   *
   * @param execRootParent the execution root's parent path. This will be [output_base]/execroot.
   */
  public ArtifactFactory(Path execRootParent, String derivedPathPrefix) {
    this.execRootParent = execRootParent;
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

  public void setSiblingRepositoryLayout(boolean siblingRepositoryLayout) {
    this.siblingRepositoryLayout = siblingRepositoryLayout;
  }

  /**
   * Set the set of known packages and their corresponding source artifact roots. Must be called
   * exactly once after construction or clear().
   *
   * @param packageRoots provider of a source root given a package identifier.
   */
  public synchronized void setPackageRoots(PackageRoots.PackageRootLookup packageRoots) {
    this.packageRoots = packageRoots;
  }

  public synchronized void noteAnalysisStarting() {
    sourceArtifactCache.newBuild();
  }

  @Override
  public SourceArtifact getSourceArtifact(PathFragment execPath, Root root, ArtifactOwner owner) {
    Preconditions.checkArgument(
        execPath.isAbsolute() == root.isAbsolute(), "%s %s %s", execPath, root, owner);
    Preconditions.checkNotNull(owner, "%s %s", execPath, root);
    Preconditions.checkNotNull(
        sourceArtifactRoots, "Not initialized for %s %s %s", execPath, root, owner);
    return (SourceArtifact)
        getArtifact(
            Preconditions.checkNotNull(
                sourceArtifactRoots.get(root),
                "%s has no ArtifactRoot (%s) in %s",
                root,
                execPath,
                sourceArtifactRoots),
            execPath,
            owner,
            null,
            /*contentBasedPath=*/ false);
  }

  @Override
  public SourceArtifact getSourceArtifact(PathFragment execPath, Root root) {
    return getSourceArtifact(execPath, root, ArtifactOwner.NULL_OWNER);
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
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    return getDerivedArtifact(rootRelativePath, root, owner, /*contentBasedPath=*/ false);
  }

  /**
   * Same as {@link #getDerivedArtifact(PathFragment, ArtifactRoot, ArtifactOwner)} but includes the
   * option to use a content-based path for this artifact (see {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfiguration#useContentBasedOutputPaths}).
   */
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath,
      ArtifactRoot root,
      ArtifactOwner owner,
      boolean contentBasedPath) {
    validatePath(rootRelativePath, root);
    return (Artifact.DerivedArtifact)
        getArtifact(
            root, root.getExecPath().getRelative(rootRelativePath), owner, null, contentBasedPath);
  }

  /**
   * Returns an artifact that represents the output directory of a Fileset at the given
   * root-relative path under the given root, creating it if not found. This method only works for
   * normalized, relative paths.
   *
   * <p>The root must be below the execRootParent, and the execPath of the resulting Artifact is
   * computed as {@code root.getRelative(rootRelativePath).relativeTo(root.execRoot)}.
   */
  public Artifact.DerivedArtifact getFilesetArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return (Artifact.DerivedArtifact)
        getArtifact(
            root,
            root.getExecPath().getRelative(rootRelativePath),
            owner,
            SpecialArtifactType.FILESET,
            /*contentBasedPath=*/ false);
  }

  /**
   * Returns an artifact that represents a TreeArtifact; that is, a directory containing some tree
   * of ArtifactFiles unknown at analysis time.
   *
   * <p>The root must be below the execRootParent, and the execPath of the resulting Artifact is
   * computed as {@code root.getRelative(rootRelativePath).relativeTo(root.execRoot)}.
   */
  public Artifact.SpecialArtifact getTreeArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return (Artifact.SpecialArtifact)
        getArtifact(
            root,
            root.getExecPath().getRelative(rootRelativePath),
            owner,
            SpecialArtifactType.TREE,
            /*contentBasedPath=*/ false);
  }

  /**
   * Returns an artifact that represents an unresolved symlink; that is, an artifact whose value is
   * a symlink and is never dereferenced.
   *
   * <p>The root must be below the execRootParent, and the execPath of the resulting Artifact is
   * computed as {@code root.getRelative(rootRelativePath).relativeTo(root.execRoot)}.
   */
  public Artifact.SpecialArtifact getSymlinkArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return (Artifact.SpecialArtifact)
        getArtifact(
            root,
            root.getExecPath().getRelative(rootRelativePath),
            owner,
            SpecialArtifactType.UNRESOLVED_SYMLINK,
            /*contentBasedPath=*/ false);
  }

  public Artifact.DerivedArtifact getConstantMetadataArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return (Artifact.DerivedArtifact)
        getArtifact(
            root,
            root.getExecPath().getRelative(rootRelativePath),
            owner,
            SpecialArtifactType.CONSTANT_METADATA,
            /*contentBasedPath=*/ false);
  }

  /**
   * Returns the Artifact for the specified path, creating one if not found and setting the <code>
   * root</code> and <code>execPath</code> to the specified values.
   */
  private Artifact getArtifact(
      ArtifactRoot root,
      PathFragment execPath,
      ArtifactOwner owner,
      @Nullable SpecialArtifactType type,
      boolean contentBasedPath) {
    Preconditions.checkNotNull(root);
    Preconditions.checkNotNull(execPath);

    if (!root.isSourceRoot()) {
      return createArtifact(root, execPath, owner, type, contentBasedPath);
    }

    // Double-checked locking to avoid locking cost when possible.
    SourceArtifact firstArtifact = sourceArtifactCache.getArtifact(execPath);
    if (firstArtifact != null && !firstArtifact.differentOwnerOrRoot(owner, root)) {
      return firstArtifact;
    }
    SourceArtifactCache.Entry newEntry =
        sourceArtifactCache.pathToSourceArtifact.compute(
            execPath,
            (k, entry) -> {
              if (entry == null
                  || entry.getArtifact() == null
                  || entry.getArtifact().differentOwnerOrRoot(owner, root)) {
                // There really should be a safety net that makes it impossible to create two
                // Artifacts with the same exec path but a different Owner, but we also need to
                // reuse Artifacts from previous builds.
                return sourceArtifactCache
                .new Entry(
                    (SourceArtifact)
                        createArtifact(root, execPath, owner, type, /*contentBasedPath=*/ false));
              }
              return entry;
            });
    return newEntry.getArtifact();
  }

  private Artifact createArtifact(
      ArtifactRoot root,
      PathFragment execPath,
      ArtifactOwner owner,
      @Nullable SpecialArtifactType type,
      boolean contentBasedPath) {
    Preconditions.checkNotNull(owner);
    if (type == null) {
      return root.isSourceRoot()
          ? new Artifact.SourceArtifact(root, execPath, owner)
          : new Artifact.DerivedArtifact(root, execPath, (ActionLookupKey) owner, contentBasedPath);
    } else {
      return new Artifact.SpecialArtifact(root, execPath, (ActionLookupKey) owner, type);
    }
  }

  /**
   * Returns an {@link Artifact} with exec path formed by composing {@code baseExecPath} and {@code
   * relativePath} (via {@code baseExecPath.getRelative(relativePath)} if baseExecPath is not null).
   * That Artifact will have root determined by the package roots of this factory if it lives in a
   * subpackage distinct from that of baseExecPath, and {@code baseRoot} otherwise.
   *
   * <p>Thread-safety: does only reads until the call to #createArtifactIfNotValid. That may perform
   * mutations, but is thread-safe. There is the potential for a race in which one thread observes
   * no matching artifact in {@link #sourceArtifactCache} initially, but when it goes to create it,
   * does find it there, but that is a benign race.
   */
  @ThreadSafe
  public Artifact resolveSourceArtifactWithAncestor(
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

    // Source exec paths cannot escape the source root.
    if (siblingRepositoryLayout) {
      // The exec path may start with .. if using --experimental_sibling_repository_layout, so test
      // the subfragment from index 1 onwards.
      if (execPath.subFragment(1).containsUplevelReferences()) {
        return null;
      }
    } else if (execPath.containsUplevelReferences()) {
      return null;
    }

    // Don't create an artifact if it's derived.
    if (isDerivedArtifact(execPath)) {
      return null;
    }
    Artifact artifact = sourceArtifactCache.getArtifactIfValid(execPath);
    if (artifact != null) {
      return artifact;
    }
    Root sourceRoot =
        findSourceRoot(
            execPath, baseExecPath, baseRoot == null ? null : baseRoot.getRoot(), repositoryName);
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

    Pair<RepositoryName, PathFragment> repo =
        RepositoryName.fromPathFragment(dir, siblingRepositoryLayout);
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
  public Map<PathFragment, Artifact> resolveSourceArtifacts(
      Iterable<PathFragment> execPaths, PackageRootResolver resolver) throws InterruptedException {
    Map<PathFragment, Artifact> result = new HashMap<>();
    ArrayList<PathFragment> unresolvedPaths = new ArrayList<>();

    for (PathFragment execPath : execPaths) {
      if (execPath.containsUplevelReferences()) {
        // Source exec paths cannot escape the source root.
        result.put(execPath, null);
        continue;
      }
      if (isDerivedArtifact(execPath)) {
        result.put(execPath, null);
      } else {
        // First try a quick map lookup to see if the artifact already exists.
        Artifact a = sourceArtifactCache.getArtifactIfValid(execPath);
        if (a != null) {
          result.put(execPath, a);
        } else {
          // Remember this path, maybe we can resolve it with the help of PackageRootResolver.
          unresolvedPaths.add(execPath);
        }
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
  public Path getPathFromSourceExecPath(Path execRoot, PathFragment execPath) {
    Preconditions.checkState(
        !execPath.startsWith(derivedPathPrefix), "%s is derived: %s", execPath, derivedPathPrefix);
    Root sourceRoot =
        packageRoots.getRootForPackage(PackageIdentifier.create(RepositoryName.MAIN, execPath));
    if (sourceRoot != null) {
      return sourceRoot.getRelative(execPath);
    }
    return execRoot.getRelative(execPath);
  }

  @ThreadSafe
  private Artifact createArtifactIfNotValid(Root sourceRoot, PathFragment execPath) {
    if (sourceRoot == null) {
      return null;  // not a path that we can find...
    }
    Artifact artifact = sourceArtifactCache.getArtifact(execPath);
    if (artifact != null && sourceRoot.equals(artifact.getRoot().getRoot())) {
      // Source root of existing artifact hasn't changed so we should mark corresponding entry in
      // the cache as valid.
      sourceArtifactCache.pathToSourceArtifact.compute(
          execPath,
          (k, cacheEntry) -> {
            SourceArtifact validArtifact = cacheEntry.getArtifact();
            if (!cacheEntry.isArtifactValid()) {
              // Wasn't previously known to be valid.
              return sourceArtifactCache.new Entry(validArtifact);
            }
            Preconditions.checkState(
                artifact.equals(validArtifact),
                "Mismatched artifacts: %s %s",
                artifact,
                validArtifact);
            return cacheEntry;
          });
      return artifact;
    } else {
      // Must be a new artifact or artifact in the cache is stale, so create a new one.
      return getSourceArtifact(execPath, sourceRoot, ArtifactOwner.NULL_OWNER);
    }
  }

  /**
   * Determines if an artifact is derived, that is, its root is a derived root or its exec path
   * starts with the bazel-out prefix.
   *
   * @param execPath The artifact's exec path.
   */
  @VisibleForTesting // for our own unit tests only.
  boolean isDerivedArtifact(PathFragment execPath) {
    return execPath.startsWith(derivedPathPrefix);
  }
}
