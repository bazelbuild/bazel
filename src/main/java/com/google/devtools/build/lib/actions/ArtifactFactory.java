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

import static java.util.Comparator.comparing;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.BiFunction;
import javax.annotation.Nullable;

/** A cache of Artifacts, keyed by Path. */
@ThreadSafe
public class ArtifactFactory implements ArtifactResolver {

  private final Path execRootParent;
  private final Path externalSourceBase;
  private final PathFragment derivedPathPrefix;
  private boolean siblingRepositoryLayout = false;

  /** Cache of source artifacts. */
  private final SourceArtifactCache sourceArtifactCache = new SourceArtifactCache();

  /**
   * Map of package names to source root paths so that we can create source artifact paths given
   * execPaths in the symlink forest.
   */
  private PackageRoots.PackageRootLookup packageRoots;

  private static class SourceArtifactCache {

    private record Entry(SourceArtifact artifact, int buildId) {
      boolean isInvalid(int currentBuildId) {
        return buildId != currentBuildId;
      }
    }

    /**
     * The main Path to source artifact cache. There will always be exactly one canonical artifact
     * for a given source path.
     *
     * <p>Since some use cases require case-insensitive lookups, the map uses a case-insensitive key
     * lookup. A ConcurrentSkipListMap supports this without a PathFragment wrapper, which saves
     * memory. The corresponding value is either a single Entry, or a list of Entry objects if there
     * are multiple artifacts with case-insensitively equivalent paths. This structure is heavily
     * optimized for the common case of a single artifact per case-insensitive equivalence class and
     * may perform poorly if there are many artifacts with case-insensitively equivalent paths.
     */
    private final ConcurrentMap<PathFragment, Object /* Entry | CopyOnWriteArrayList<Entry> */>
        pathToSourceArtifact =
            new ConcurrentSkipListMap<>(
                comparing(
                    pathFragment -> StringEncoding.internalToUnicode(pathFragment.getPathString()),
                    String.CASE_INSENSITIVE_ORDER));

    /** Id of current build. Has to be increased every time before analysis starts. */
    private int buildId = -1;

    @Nullable
    private Entry unwrapCacheObject(PathFragment execPath, Object cacheObject) {
      return switch (cacheObject) {
        case null -> null;
        case Entry entry -> entry.artifact().getExecPath().equals(execPath) ? entry : null;
        case CopyOnWriteArrayList<?> entries -> {
          for (Object entryObject : entries) {
            var entry = (Entry) entryObject;
            if (entry.artifact().getExecPath().equals(execPath)) {
              yield entry;
            }
          }
          yield null;
        }
        default ->
            throw new IllegalStateException(
                "Unexpected cache object type: %s, value: %s"
                    .formatted(cacheObject.getClass(), cacheObject));
      };
    }

    @Nullable
    private Entry getEntry(PathFragment execPath) {
      return unwrapCacheObject(execPath, pathToSourceArtifact.get(execPath));
    }

    @SuppressWarnings("unchecked")
    private Entry computeEntry(
        PathFragment execPath, BiFunction<PathFragment, Entry, Entry> computeFunction) {
      return unwrapCacheObject(
          execPath,
          pathToSourceArtifact.compute(
              execPath,
              (key, cacheObject) ->
                  switch (cacheObject) {
                    // No entry for this case-insensitive path, thus also not for this exact casing.
                    case null -> computeFunction.apply(execPath, null);
                    // The lookup was case-insensitive, so the single cache entry may not be valid
                    // for this exact casing. If it isn't, switch to a list.
                    case Entry entry ->
                        entry.artifact().getExecPath().equals(execPath)
                            ? computeFunction.apply(execPath, entry)
                            : new CopyOnWriteArrayList<>(
                                new Entry[] {entry, computeFunction.apply(execPath, null)});
                    case CopyOnWriteArrayList<?> rawEntries -> {
                      var entries = (CopyOnWriteArrayList<Entry>) rawEntries;
                      for (Entry entry : entries) {
                        // Update the existing entry for this exact casing if it exists.
                        if (entry.artifact().getExecPath().equals(execPath)) {
                          Entry newEntry = computeFunction.apply(execPath, entry);
                          if (newEntry != entry) {
                            entries.set(entries.indexOf(entry), newEntry);
                          }
                          yield entries;
                        }
                      }
                      // No entry for this exact casing, add a new one.
                      entries.add(computeFunction.apply(execPath, null));
                      yield entries;
                    }
                    default ->
                        throw new IllegalStateException(
                            "Unexpected cache object type: %s, value: %s"
                                .formatted(cacheObject.getClass(), cacheObject));
                  }));
    }

    /** Returns artifact if it present in the cache, otherwise null. */
    @Nullable
    @ThreadSafe
    SourceArtifact getArtifact(PathFragment execPath) {
      Entry cacheEntry = getEntry(execPath);
      return cacheEntry == null ? null : cacheEntry.artifact();
    }

    /**
     * Returns artifact if it is present in the cache and has been verified to be valid for this
     * build, otherwise null. Note that if the artifact's package is not part of the current build,
     * our differing methods of validating source roots (via {@link PackageRootResolver} and via
     * {@link #findSourceRoot}) may disagree. In that case, the artifact will be valid, but unusable
     * by any action (since no action has properly declared it as an input).
     */
    @Nullable
    @ThreadSafe
    SourceArtifact getArtifactIfValid(PathFragment execPath) {
      Entry cacheEntry = getEntry(execPath);
      return (cacheEntry == null || cacheEntry.isInvalid(buildId)) ? null : cacheEntry.artifact();
    }

    /**
     * Returns all entries with case-insensitively equivalent exec paths. The returned list contains
     * the raw cache entries, which may or may not be valid for the current build.
     */
    @SuppressWarnings("unchecked")
    @ThreadSafe
    private ImmutableList<Entry> getEntriesWithAsciiCaseInsensitivePath(PathFragment execPath) {
      Object cacheObject = pathToSourceArtifact.get(execPath);
      return switch (cacheObject) {
        case null -> ImmutableList.of();
        case Entry entry -> ImmutableList.of(entry);
        case CopyOnWriteArrayList<?> entries ->
            ImmutableList.copyOf((CopyOnWriteArrayList<Entry>) entries);
        default ->
            throw new IllegalStateException(
                "Unexpected cache object type: %s, value: %s"
                    .formatted(cacheObject.getClass(), cacheObject));
      };
    }

    /**
     * Returns a list of artifacts with case-insensitively equivalent exec paths that are present in
     * the cache and have been verified to be valid for this build. Note that if the artifacts'
     * packages are not part of the current build, our differing methods of validating source roots
     * (via {@link PackageRootResolver} and via {@link #findSourceRoot}) may disagree. In that case,
     * the artifacts will be valid, but unusable by any action (since no action has properly
     * declared them as inputs).
     */
    @ThreadSafe
    ImmutableList<SourceArtifact> getValidArtifactsWithAsciiCaseInsensitivePath(
        PathFragment execPath) {
      return getEntriesWithAsciiCaseInsensitivePath(execPath).stream()
          .filter(entry -> !entry.isInvalid(buildId))
          .map(Entry::artifact)
          .collect(ImmutableList.toImmutableList());
    }

    /**
     * Returns a list of all artifacts with case-insensitively equivalent exec paths that are
     * present in the cache, regardless of whether they have been verified to be valid for this
     * build. This is used to find stale artifacts from previous builds that can be revalidated
     * using their original (correct-casing) exec paths.
     */
    @ThreadSafe
    ImmutableList<SourceArtifact> getAllArtifactsWithAsciiCaseInsensitivePath(
        PathFragment execPath) {
      return getEntriesWithAsciiCaseInsensitivePath(execPath).stream()
          .map(Entry::artifact)
          .collect(ImmutableList.toImmutableList());
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
    this.externalSourceBase =
        execRootParent
            .getParentDirectory()
            .getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
    this.derivedPathPrefix = PathFragment.create(derivedPathPrefix);
  }

  /** Clear the cache. */
  public synchronized void clear() {
    packageRoots = null;
    sourceArtifactCache.clear();
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
  public SourceArtifact getSourceArtifact(PathFragment execPath, Root root) {
    return getSourceArtifact(execPath, root, ArtifactOwner.NULL_OWNER);
  }

  @Override
  public SourceArtifact getSourceArtifact(PathFragment execPath, Root root, ArtifactOwner owner) {
    // TODO(jungjw): Come up with a more reliable way to distinguish external source roots.
    ArtifactRoot artifactRoot =
        root.asPath() != null && root.asPath().startsWith(externalSourceBase)
            ? ArtifactRoot.asExternalSourceRoot(root)
            : ArtifactRoot.asSourceRoot(root);
    return getSourceArtifact(execPath, artifactRoot, owner);
  }

  public SourceArtifact getSourceArtifact(
      PathFragment execPath, ArtifactRoot root, ArtifactOwner owner) {
    Preconditions.checkArgument(
        execPath.isAbsolute() == root.getRoot().isAbsolute(), "%s %s %s", execPath, root, owner);
    Preconditions.checkNotNull(owner, "%s %s", execPath, root);
    return (SourceArtifact) getArtifact(root, execPath, owner, /* type= */ null);
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
    // Preconditions.checkArgument(derivedRoots.contains(root), "%s not in %s", root, derivedRoots);
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
    validatePath(rootRelativePath, root);
    return (Artifact.DerivedArtifact)
        getArtifact(root, root.getExecPath().getRelative(rootRelativePath), owner, null);
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
            SpecialArtifactType.FILESET);
  }

  public Artifact.DerivedArtifact getRunfilesArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return (Artifact.DerivedArtifact)
        getArtifact(
            root,
            root.getExecPath().getRelative(rootRelativePath),
            owner,
            SpecialArtifactType.RUNFILES);
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
            SpecialArtifactType.TREE);
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
            SpecialArtifactType.UNRESOLVED_SYMLINK);
  }

  public Artifact.DerivedArtifact getConstantMetadataArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    validatePath(rootRelativePath, root);
    return (Artifact.DerivedArtifact)
        getArtifact(
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
    SourceArtifact firstArtifact = sourceArtifactCache.getArtifact(execPath);
    if (firstArtifact != null && !firstArtifact.differentOwnerOrRoot(owner, root)) {
      return firstArtifact;
    }
    SourceArtifactCache.Entry newEntry =
        sourceArtifactCache.computeEntry(
            execPath,
            (k, entry) -> {
              if (entry == null
                  || entry.artifact() == null
                  || entry.artifact().differentOwnerOrRoot(owner, root)) {
                // There really should be a safety net that makes it impossible to create two
                // Artifacts with the same exec path but a different Owner, but we also need to
                // reuse Artifacts from previous builds.
                return new SourceArtifactCache.Entry(
                    (SourceArtifact) createArtifact(root, execPath, owner, type),
                    sourceArtifactCache.buildId);
              }
              return entry;
            });
    return newEntry.artifact();
  }

  private static Artifact createArtifact(
      ArtifactRoot root,
      PathFragment execPath,
      ArtifactOwner owner,
      @Nullable SpecialArtifactType type) {
    Preconditions.checkNotNull(owner);
    if (type == null) {
      return root.isSourceRoot()
          ? new Artifact.SourceArtifact(root, execPath, owner)
          : DerivedArtifact.create(root, execPath, (ActionLookupKey) owner);
    } else {
      return SpecialArtifact.create(root, execPath, (ActionLookupKey) owner, type);
    }
  }

  private boolean isDefinitelyNotSourceExecPath(PathFragment execPath) {
    // Source exec paths cannot escape the source root.
    if (siblingRepositoryLayout) {
      // The exec path may start with .. if using --experimental_sibling_repository_layout, so test
      // the subfragment from index 1 onwards.
      if (execPath.subFragment(1).containsUplevelReferences()) {
        return true;
      }
    } else if (execPath.containsUplevelReferences()) {
      return true;
    }

    return false;
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
  @Nullable
  @ThreadSafe
  public SourceArtifact resolveSourceArtifactWithAncestor(
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

    if (isDefinitelyNotSourceExecPath(execPath)) {
      return null;
    }

    // Don't create an artifact if it's derived.
    if (isDerivedArtifact(execPath)) {
      return null;
    }
    SourceArtifact artifact = sourceArtifactCache.getArtifactIfValid(execPath);
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
  public SourceArtifact resolveSourceArtifact(
      PathFragment execPath, RepositoryName repositoryName) {
    return resolveSourceArtifactWithAncestor(execPath, null, null, repositoryName);
  }

  @Override
  public ImmutableList<SourceArtifact> resolveSourceArtifactsAsciiCaseInsensitively(
      PathFragment execPath, RepositoryName repositoryName) {
    if (isDefinitelyNotSourceExecPath(execPath)) {
      return ImmutableList.of();
    }

    // Don't create an artifact if it's derived.
    if (isDerivedArtifact(execPath)) {
      return ImmutableList.of();
    }
    var artifacts = sourceArtifactCache.getValidArtifactsWithAsciiCaseInsensitivePath(execPath);
    if (!artifacts.isEmpty()) {
      return artifacts;
    }
    // The case-insensitive cache may have artifacts from a previous build that aren't valid yet.
    // Try to revalidate them using their original (correct-casing) exec paths before falling back
    // to creating a new artifact with the queried (potentially wrong-casing) exec path.
    var staleArtifacts =
        sourceArtifactCache.getAllArtifactsWithAsciiCaseInsensitivePath(execPath);
    if (!staleArtifacts.isEmpty()) {
      var revalidated = ImmutableList.<SourceArtifact>builder();
      for (SourceArtifact stale : staleArtifacts) {
        Root sourceRoot =
            findSourceRoot(
                stale.getExecPath(),
                /* baseExecPath= */ null,
                /* baseRoot= */ null,
                repositoryName);
        SourceArtifact valid = createArtifactIfNotValid(sourceRoot, stale.getExecPath());
        if (valid != null) {
          revalidated.add(valid);
        }
      }
      var result = revalidated.build();
      if (!result.isEmpty()) {
        return result;
      }
    }
    Root sourceRoot =
        findSourceRoot(execPath, /* baseExecPath= */ null, /* baseRoot= */ null, repositoryName);
    SourceArtifact newArtifact = createArtifactIfNotValid(sourceRoot, execPath);
    if (newArtifact == null) {
      return ImmutableList.of();
    }
    return ImmutableList.of(newArtifact);
  }

  @Nullable
  @Override
  public Map<PathFragment, SourceArtifact> resolveSourceArtifacts(
      Iterable<PathFragment> execPaths, PackageRootResolver resolver)
      throws PackageRootResolver.PackageRootException, InterruptedException {
    Map<PathFragment, SourceArtifact> result = new HashMap<>();
    ArrayList<PathFragment> unresolvedPaths = new ArrayList<>();

    for (PathFragment execPath : execPaths) {
      if (isDefinitelyNotSourceExecPath(execPath)) {
        result.put(execPath, null);
        continue;
      }
      if (isDerivedArtifact(execPath)) {
        result.put(execPath, null);
      } else {
        // First try a quick map lookup to see if the artifact already exists.
        SourceArtifact a = sourceArtifactCache.getArtifactIfValid(execPath);
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

  @Nullable
  @ThreadSafe
  private SourceArtifact createArtifactIfNotValid(Root sourceRoot, PathFragment execPath) {
    if (sourceRoot == null) {
      return null; // not a path that we can find...
    }
    SourceArtifact artifact = sourceArtifactCache.getArtifact(execPath);
    if (artifact != null && sourceRoot.equals(artifact.getRoot().getRoot())) {
      // Source root of existing artifact hasn't changed so we should mark corresponding entry in
      // the cache as valid.
      sourceArtifactCache.computeEntry(
          execPath,
          (k, cacheEntry) -> {
            SourceArtifact validArtifact = cacheEntry.artifact();
            if (cacheEntry.isInvalid(sourceArtifactCache.buildId)) {
              // Wasn't previously known to be valid.
              return new SourceArtifactCache.Entry(validArtifact, sourceArtifactCache.buildId);
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

  @Override
  public boolean isDerivedArtifact(PathFragment execPath) {
    return execPath.startsWith(derivedPathPrefix);
  }
}
