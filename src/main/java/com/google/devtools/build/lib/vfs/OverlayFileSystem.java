// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.common.collect.ImmutableSortedSet;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * A file system that combines multiple underlying sources (other file systems or in-memory
 * metadata) into a single unified view, correctly resolving symlink chains that cross sources.
 *
 * <p>Since a symlink and its target may reside in different sources, with an arbitrary number of
 * indirections in between, no single source can resolve a symlink chain by itself. This class
 * therefore implements path-sensitive operations in a canonicalize-first style: the path is first
 * made canonical against the <i>combined</i> view of all sources, by repeatedly invoking the
 * {@link #statNofollow} and {@link #readSymlinkNofollow} primitives provided by the subclass, and
 * only then is the terminal operation performed on the canonical path, again via a subclass hook
 * that must not follow symlinks. Operations that read or write file contents remain the
 * responsibility of subclasses, which should use {@link #canonicalize} or {@link
 * #canonicalizeParent} to apply the same discipline to them. Canonicalization results are cached
 * in a {@link PathCanonicalizer} trie that is invalidated on structure-changing operations; see
 * {@link #mayCacheResolution} for the caching contract.
 *
 * <p>Subclasses define how the sources are combined: which sources are consulted for a given path,
 * in which order, and which sources writes are applied to. The hooks receive paths whose parent
 * (and, where documented, final segment) has already been canonicalized, so they must dispatch to
 * the underlying sources without following symlinks.
 *
 * <p>Subclasses for which only part of the file system namespace is overlaid can bypass the
 * canonicalize-first machinery for paths outside that part by returning a delegate file system
 * from {@link #wholesaleDelegate}.
 *
 * <p>As with the individual sources, the result of concurrent operations on paths whose structure
 * (symlinks and directories) is being modified concurrently is undefined.
 */
public abstract class OverlayFileSystem extends FileSystem implements PathCanonicalizer.Resolver {

  private final PathCanonicalizer pathCanonicalizer =
      new PathCanonicalizer(this, this::mayCacheResolution);

  protected OverlayFileSystem(DigestHashFunction digestFunction) {
    super(digestFunction);
  }

  /**
   * Returns whether the canonicalization of the given path, all but the last segment of which is
   * canonical, may be cached until invalidated through a method of this class.
   *
   * <p>Subclasses must return false for paths whose symlink structure can change through channels
   * that bypass this file system (e.g. external processes writing directly to an underlying
   * source), as such changes don't invalidate the cache. The resolution of these paths, as well as
   * of all paths underneath them, is recomputed on every operation.
   */
  protected boolean mayCacheResolution(PathFragment path) {
    return true;
  }

  /**
   * Returns a file system to which operations on the given path are delegated wholesale, bypassing
   * the overlay machinery including symlink canonicalization, or null if the path must be handled
   * by the overlay.
   *
   * <p>If a symlink chain starting at an overlay-handled path escapes into wholesale-delegated
   * territory, the overlay hooks are still invoked for the escaped part of the chain and for the
   * resulting canonical path; hooks must handle such paths, typically by routing them to this
   * delegate. Conversely, a symlink chain starting at a wholesale-delegated path is resolved
   * entirely by the delegate, even if it points back into overlay-handled territory; subclasses
   * must choose a delegation boundary for which this is acceptable.
   */
  @Nullable
  protected FileSystem wholesaleDelegate(PathFragment path) {
    return null;
  }

  /**
   * Stats a path in the combined view of all sources without following symlinks.
   *
   * <p>All but the last segment of the path are guaranteed to be canonical.
   *
   * @return the file status, or null if the path does not exist in any source (including if a
   *     non-directory is encountered where a directory was expected)
   * @throws IOException if an error other than file not found occurred
   */
  @Nullable
  protected abstract FileStatus statNofollow(PathFragment path) throws IOException;

  /**
   * Reads a symlink in the combined view of all sources.
   *
   * <p>All but the last segment of the path are guaranteed to be canonical.
   *
   * @throws NotASymlinkException if the path exists but is not a symlink
   * @throws IOException if the symlink could not be read for any other reason
   */
  protected abstract PathFragment readSymlinkNofollow(PathFragment path) throws IOException;

  /**
   * Reads the entries of a directory in the combined view of all sources, without following
   * symlinks when determining entry types.
   *
   * <p>The path is guaranteed to be canonical. The returned entries must have distinct names, but
   * may be in any order.
   *
   * @throws FileNotFoundException if the directory does not exist in any source
   * @throws IOException if the directory could not be read for any other reason
   */
  protected abstract Collection<Dirent> readdirNofollow(PathFragment path) throws IOException;

  /**
   * Returns the digest of the file denoted by the given canonical path, computing it if necessary.
   */
  protected abstract byte[] getDigestNofollow(PathFragment path) throws IOException;

  /**
   * Returns the digest of the file denoted by the given canonical path if it can be obtained
   * cheaply, otherwise null.
   */
  @Nullable
  protected abstract byte[] getFastDigestNofollow(PathFragment path) throws IOException;

  /**
   * Deletes the entry denoted by the given path (a symlink itself, not its target) from all
   * sources that contain it, returning true if any source contained it.
   *
   * <p>All but the last segment of the path are guaranteed to be canonical. The canonicalization
   * cache has already been invalidated.
   */
  protected abstract boolean deleteNofollow(PathFragment path) throws IOException;

  /**
   * Renames an entry, applying the change to sources as defined by the subclass.
   *
   * <p>All but the last segment of both paths are guaranteed to be canonical. The canonicalization
   * cache has already been invalidated for both paths.
   */
  protected abstract void renameToNofollow(PathFragment sourcePath, PathFragment targetPath)
      throws IOException;

  /**
   * Creates a symlink in the appropriate source(s).
   *
   * <p>All but the last segment of the link path are guaranteed to be canonical.
   */
  protected abstract void createSymbolicLinkNofollow(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
      throws IOException;

  /** Sets the last modified time of the entry denoted by the given canonical path. */
  protected abstract void setLastModifiedTimeNofollow(PathFragment path, long newTime)
      throws IOException;

  /** Returns whether the entry denoted by the given canonical path is readable. */
  protected abstract boolean isReadableNofollow(PathFragment path) throws IOException;

  /** Returns whether the entry denoted by the given canonical path is writable. */
  protected abstract boolean isWritableNofollow(PathFragment path) throws IOException;

  /** Returns whether the entry denoted by the given canonical path is executable. */
  protected abstract boolean isExecutableNofollow(PathFragment path) throws IOException;

  /** Sets the readable bit of the entry denoted by the given canonical path. */
  protected abstract void setReadableNofollow(PathFragment path, boolean readable)
      throws IOException;

  /** Sets the writable bit of the entry denoted by the given canonical path. */
  protected abstract void setWritableNofollow(PathFragment path, boolean writable)
      throws IOException;

  /** Sets the executable bit of the entry denoted by the given canonical path. */
  protected abstract void setExecutableNofollow(PathFragment path, boolean executable)
      throws IOException;

  /** Sets the permissions of the entry denoted by the given canonical path. */
  protected abstract void chmodNofollow(PathFragment path, int mode) throws IOException;

  /** Canonicalizes the entire path (the equivalent of {@link Symlinks#FOLLOW}). */
  protected final PathFragment canonicalize(PathFragment path) throws IOException {
    return pathCanonicalizer.resolveSymbolicLinks(path);
  }

  /**
   * Canonicalizes all but the last segment of a path (the equivalent of {@link Symlinks#NOFOLLOW}).
   */
  protected final PathFragment canonicalizeParent(PathFragment path) throws IOException {
    PathFragment parent = path.getParentDirectory();
    if (parent == null) {
      return path;
    }
    return canonicalize(parent).getChild(path.getBaseName());
  }

  /**
   * Invalidates cached canonicalization results for all paths underneath the given path prefix.
   *
   * <p>Must be called by subclasses whenever they change the structure of the file system (i.e.,
   * create, move or delete paths) other than through methods of this class.
   */
  protected final void invalidatePrefix(PathFragment prefix) {
    pathCanonicalizer.clearPrefix(prefix);
  }

  /** Invalidates all cached canonicalization results. */
  protected final void clearCanonicalizationCache() {
    pathCanonicalizer.clear();
  }

  @Override
  public final Path resolveSymbolicLinks(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      // Ensure that the returned path doesn't leave the overlay file system.
      return getPath(delegate.resolveSymbolicLinks(path).asFragment());
    }
    return getPath(canonicalize(path));
  }

  @Override
  @Nullable
  public final PathFragment resolveOneLink(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.resolveOneLink(path);
    }
    // The base implementation attempts to readSymbolicLink first and falls back to stat, but that
    // unnecessarily allocates a NotASymlinkException in the overwhelmingly likely non-symlink case.
    // It's more efficient to stat unconditionally.
    //
    // The parent path has already been canonicalized, so it is safe to call the *Nofollow hooks.
    var stat = statNofollow(path);
    if (stat == null) {
      throw new FileNotFoundException(path.getPathString() + " (No such file or directory)");
    }
    return stat.isSymbolicLink() ? readSymlinkNofollow(path) : null;
  }

  @Override
  public final FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.stat(path, followSymlinks);
    }
    FileStatus stat = statIfFound(path, followSymlinks);
    if (stat == null) {
      throw new FileNotFoundException(path.getPathString() + " (No such file or directory)");
    }
    return stat;
  }

  @Override
  @Nullable
  public final FileStatus statIfFound(PathFragment path, boolean followSymlinks)
      throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.statIfFound(path, followSymlinks);
    }
    PathFragment resolvedPath;
    try {
      resolvedPath = followSymlinks ? canonicalize(path) : canonicalizeParent(path);
    } catch (FileNotFoundException e) {
      return null;
    }
    return statNofollow(resolvedPath);
  }

  @Override
  @Nullable
  public final FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.statNullable(path, followSymlinks);
    }
    try {
      return statIfFound(path, followSymlinks);
    } catch (IOException e) {
      return null;
    }
  }

  @Override
  public final boolean exists(PathFragment path, boolean followSymlinks) {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.exists(path, followSymlinks);
    }
    try {
      return statIfFound(path, followSymlinks) != null;
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  public final long getLastModifiedTime(PathFragment path, boolean followSymlinks)
      throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getLastModifiedTime(path, followSymlinks);
    }
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  public final long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getFileSize(path, followSymlinks);
    }
    return stat(path, followSymlinks).getSize();
  }

  @Override
  public final PathFragment readSymbolicLink(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.readSymbolicLink(path);
    }
    return readSymlinkNofollow(canonicalizeParent(path));
  }

  @Override
  public final PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.readSymbolicLinkUnchecked(path);
    }
    return readSymlinkNofollow(canonicalizeParent(path));
  }

  @Override
  public final Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getDirectoryEntries(path);
    }
    return getDirectoryContents(path, /* followSymlinks= */ false, Dirent::getName);
  }

  @Override
  public final Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
      throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.readdir(path, followSymlinks);
    }
    return getDirectoryContents(path, followSymlinks, Function.identity());
  }

  private <T extends Comparable<T>> ImmutableSortedSet<T> getDirectoryContents(
      PathFragment path, boolean followSymlinks, Function<Dirent, T> transformer)
      throws IOException {
    PathFragment resolvedPath = canonicalize(path);
    // Sort entries to get a deterministic order.
    ImmutableSortedSet.Builder<T> builder = ImmutableSortedSet.naturalOrder();
    for (Dirent entry : readdirNofollow(resolvedPath)) {
      builder.add(
          transformer.apply(maybeFollowSymlinkForDirent(resolvedPath, entry, followSymlinks)));
    }
    return builder.build();
  }

  private Dirent maybeFollowSymlinkForDirent(
      PathFragment dirPath, Dirent entry, boolean followSymlinks) {
    if (!followSymlinks || !entry.getType().equals(Dirent.Type.SYMLINK)) {
      return entry;
    }
    PathFragment path = dirPath.getChild(entry.getName());
    FileStatus st = statNullable(path, /* followSymlinks= */ true);
    return new Dirent(entry.getName(), direntFromStat(st));
  }

  @Override
  public final byte[] getDigest(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getDigest(path);
    }
    return getDigestNofollow(canonicalize(path));
  }

  @Override
  @Nullable
  public final byte[] getFastDigest(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getFastDigest(path);
    }
    return getFastDigestNofollow(canonicalize(path));
  }

  @Override
  public final boolean delete(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      // The path may have been cached while resolving an overlay-handled symlink chain escaping
      // into wholesale-delegated territory.
      invalidatePrefix(path);
      return delegate.delete(path);
    }
    PathFragment resolvedPath;
    try {
      resolvedPath = canonicalizeParent(path);
    } catch (FileNotFoundException ignored) {
      // Failure to delete a nonexistent path is not an error.
      return false;
    }

    // The result of concurrent operations on paths whose structure is modified concurrently is
    // undefined, so there's no risk of a race condition below. Invalidate both the path as given
    // (which removes any symlink it was reached through from the cache) and the resolved path
    // (where the deleted entry is actually cached).
    invalidatePrefix(path);
    if (!resolvedPath.equals(path)) {
      invalidatePrefix(resolvedPath);
    }

    return deleteNofollow(resolvedPath);
  }

  @Override
  public final void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    var sourceDelegate = wholesaleDelegate(sourcePath);
    if (sourceDelegate != null && sourceDelegate == wholesaleDelegate(targetPath)) {
      // The paths may have been cached while resolving an overlay-handled symlink chain escaping
      // into wholesale-delegated territory.
      invalidatePrefix(sourcePath);
      invalidatePrefix(targetPath);
      sourceDelegate.renameTo(sourcePath, targetPath);
      return;
    }
    PathFragment resolvedSourcePath = canonicalizeParent(sourcePath);
    PathFragment resolvedTargetPath = canonicalizeParent(targetPath);

    // The result of concurrent operations on paths whose structure is modified concurrently is
    // undefined, so there's no risk of a race condition below. Invalidate both the paths as given
    // (which removes any symlinks they were reached through from the cache) and the resolved
    // paths (where the moved entries are actually cached).
    invalidatePrefix(sourcePath);
    invalidatePrefix(targetPath);
    if (!resolvedSourcePath.equals(sourcePath)) {
      invalidatePrefix(resolvedSourcePath);
    }
    if (!resolvedTargetPath.equals(targetPath)) {
      invalidatePrefix(resolvedTargetPath);
    }

    renameToNofollow(resolvedSourcePath, resolvedTargetPath);
  }

  @Override
  public final void createSymbolicLink(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
      throws IOException {
    var delegate = wholesaleDelegate(linkPath);
    if (delegate != null) {
      delegate.createSymbolicLink(linkPath, targetFragment, type);
      return;
    }
    createSymbolicLinkNofollow(canonicalizeParent(linkPath), targetFragment, type);
  }

  @Override
  public final void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      delegate.setLastModifiedTime(path, newTime);
      return;
    }
    setLastModifiedTimeNofollow(canonicalize(path), newTime);
  }

  @Override
  public final boolean isReadable(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.isReadable(path);
    }
    return isReadableNofollow(canonicalize(path));
  }

  @Override
  public final boolean isWritable(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.isWritable(path);
    }
    return isWritableNofollow(canonicalize(path));
  }

  @Override
  public final boolean isExecutable(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.isExecutable(path);
    }
    return isExecutableNofollow(canonicalize(path));
  }

  @Override
  public final void setReadable(PathFragment path, boolean readable) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      delegate.setReadable(path, readable);
      return;
    }
    setReadableNofollow(canonicalize(path), readable);
  }

  @Override
  public final void setWritable(PathFragment path, boolean writable) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      delegate.setWritable(path, writable);
      return;
    }
    setWritableNofollow(canonicalize(path), writable);
  }

  @Override
  public final void setExecutable(PathFragment path, boolean executable) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      delegate.setExecutable(path, executable);
      return;
    }
    setExecutableNofollow(canonicalize(path), executable);
  }

  @Override
  public final void chmod(PathFragment path, int mode) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      delegate.chmod(path, mode);
      return;
    }
    chmodNofollow(canonicalize(path), mode);
  }
}
