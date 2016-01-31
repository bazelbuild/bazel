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

package com.google.devtools.build.lib.vfs;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringTrie;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Presents a unified view of multiple virtual {@link FileSystem} instances, to which requests are
 * delegated based on a {@link PathFragment} prefix mapping.
 * If multiple prefixes apply to a given path, the *longest* (i.e. most specific) match is used.
 * The order in which the delegates are specified does not influence the mapping.
 *
 * <p>Paths are preserved absolutely, contrary to how "mount" works, e.g.:
 *    /foo/bar maps to /foo/bar on the delegate, even if it is mounted at /foo.
 *
 * <p>For example:
 * "/in" maps to InFileSystem, "/" maps to OtherFileSystem.
 * Reading from "/in/base/BUILD" through the UnionFileSystem will delegate the read operation to
 * InFileSystem, which will read "/in/base/BUILD" relative to its root.
 * ("mount" behavior would remap it to "/base/BUILD" on the delegate).
 *
 * <p>Intra-filesystem symbolic links are resolved to their ultimate targets.
 * Cross-filesystem links are not currently supported.
 */
@ThreadSafety.ThreadSafe
public class UnionFileSystem extends FileSystem {

  // Prefix trie index, allowing children to easily inherit prefix mappings
  // of their parents.
  // This does not currently handle unicode filenames.
  private StringTrie<FileSystem> pathDelegate;

  // True iff the filesystem can be modified. If false, mutating operations
  // will throw UnsupportedOperationExceptions.
  private final boolean readOnly;

  /**
   * Creates a new modifiable UnionFileSystem with prefix mappings
   * specified by a map.
   *
   * @param prefixMapping map of path prefixes to {@link FileSystem}s
   */
  public UnionFileSystem(Map<PathFragment, FileSystem> prefixMapping,
                         FileSystem rootFileSystem) {
    this(prefixMapping, rootFileSystem, /* readOnly */ false);
  }

  /**
   * Creates a new modifiable or read-only UnionFileSystem with prefix mappings
   * specified by a map.
   *
   * @param prefixMapping map of path prefixes to delegate {@link FileSystem}s
   * @param rootFileSystem root for default requests; i.e. mapping of "/"
   * @param readOnly if true, mutating operations will throw
   */
  public UnionFileSystem(Map<PathFragment, FileSystem> prefixMapping,
                         FileSystem rootFileSystem, boolean readOnly) {
    super();
    Preconditions.checkNotNull(prefixMapping);
    Preconditions.checkNotNull(rootFileSystem);
    Preconditions.checkArgument(rootFileSystem != this, "Circular root filesystem.");
    Preconditions.checkArgument(
        !prefixMapping.containsKey(PathFragment.EMPTY_FRAGMENT),
        "Attempted to specify an explicit root prefix mapping; " +
        "please use the rootFileSystem argument instead.");

    this.readOnly = readOnly;
    this.pathDelegate = new StringTrie<>();

    for (Map.Entry<PathFragment, FileSystem> prefix : prefixMapping.entrySet()) {
      FileSystem delegate = prefix.getValue();
      PathFragment prefixPath = prefix.getKey();

      // Extra slash prevents within-directory mappings, which Path can't handle.
      String path = prefixPath.getPathString();
      pathDelegate.put(path, delegate);
    }
    pathDelegate.put(PathFragment.EMPTY_FRAGMENT.getPathString(), rootFileSystem);
  }

  /**
   * Retrieves the filesystem delegate of a path mapping.
   * Does not follow symlinks (but you can call on a path preprocessed with
   * {@link #resolveSymbolicLinks} to support this use case).
   *
   * @param path the {@link Path} to map to a filesystem
   * @throws IllegalArgumentException if no delegate exists for the path
   */
  protected FileSystem getDelegate(Path path) {
    Preconditions.checkNotNull(path);

    String pathString = path.getPathString();
    FileSystem immediateDelegate = pathDelegate.get(pathString);

    // Should never actually happen if the root delegate is present.
    Preconditions.checkArgument(immediateDelegate != null, "No delegate filesystem exists for %s",
        pathString);
    return immediateDelegate;
  }

  // Associates the path with the root of the given delegate filesystem.
  // Necessary to avoid null pointer problems inside of the delegates.
  protected Path adjustPath(Path path, FileSystem delegate) {
    return delegate.getPath(path.asFragment());
  }

  /**
   * Follow a symbolic link once using the appropriate delegate filesystem, also
   * resolving parent directory symlinks.
   *
   * @param path {@link Path} to the symbolic link
   */
  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    Preconditions.checkNotNull(path);
    FileSystem delegate = getDelegate(path);
    return delegate.readSymbolicLink(adjustPath(path, delegate));
  }

  @Override
  protected PathFragment resolveOneLink(Path path) throws IOException {
    Preconditions.checkNotNull(path);
    FileSystem delegate = getDelegate(path);
    return delegate.resolveOneLink(adjustPath(path, delegate));
  }

  private void checkModifiable() {
    if (!supportsModifications()) {
      throw new UnsupportedOperationException(
          "Modifications to this " + getClass().getSimpleName() + " are disabled.");
    }
  }

  @Override
  public boolean supportsModifications() {
    return !readOnly;
  }

  @Override
  public boolean supportsSymbolicLinksNatively() {
    return true;
  }

  @Override
  public String getFileSystemType(Path path) {
    FileSystem delegate = getDelegate(path);
    return delegate.getFileSystemType(path);
  }

  @Override
  protected byte[] getMD5Digest(Path path) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.getMD5Digest(adjustPath(path, delegate));
  }

  @Override
  protected boolean createDirectory(Path path) throws IOException {
    checkModifiable();
    // When creating the exact directory that is mapped,
    // create it on both the parent's delegate and the path's delegate.
    // This is necessary both for the parent to see the directory and for the
    // delegate to use it.
    // This is present to address this problematic case:
    //   / -> RootFs
    //   /foo -> FooFs
    //   mkdir /foo
    //   ls / ("foo" would be missing if not created on the parent)
    //   ls /foo (would fail if foo weren't also present on the child)
    FileSystem delegate = getDelegate(path);
    Path parent = path.getParentDirectory();
    if (parent != null) {
      FileSystem parentDelegate = getDelegate(parent);
      if (parentDelegate != delegate) {
        // There's a possibility it already exists on the parent, so don't die
        // if the directory can't be created there.
        parentDelegate.createDirectory(adjustPath(path, parentDelegate));
      }
    }
    return delegate.createDirectory(adjustPath(path, delegate));
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.getFileSize(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected boolean delete(Path path) throws IOException {
    checkModifiable();
    FileSystem delegate = getDelegate(path);
    return delegate.delete(adjustPath(path, delegate));
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.getLastModifiedTime(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected void setLastModifiedTime(Path path, long newTime) throws IOException {
    checkModifiable();
    FileSystem delegate = getDelegate(path);
    delegate.setLastModifiedTime(adjustPath(path, delegate), newTime);
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    FileSystem delegate = getDelegate(path);
    path = adjustPath(path, delegate);
    return delegate.isSymbolicLink(path);
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    FileSystem delegate = getDelegate(path);
    return delegate.isDirectory(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    FileSystem delegate = getDelegate(path);
    return delegate.isFile(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    FileSystem delegate = getDelegate(path);
    return delegate.isSpecialFile(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    checkModifiable();
    if (!supportsSymbolicLinksNatively()) {
      throw new UnsupportedOperationException(
          "Attempted to create a symlink, but symlink support is disabled.");
    }

    FileSystem delegate = getDelegate(linkPath);
    delegate.createSymbolicLink(adjustPath(linkPath, delegate), targetFragment);
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    FileSystem delegate = getDelegate(path);
    return delegate.exists(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected FileStatus stat(final Path path, final boolean followSymlinks) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.stat(adjustPath(path, delegate), followSymlinks);
  }

  // Needs to be overridden for the delegation logic, because the
  // UnixFileSystem implements statNullable and stat as separate codepaths.
  // More generally, we wish to delegate all filesystem operations.
  @Override
  protected FileStatus statNullable(Path path, boolean followSymlinks) {
    FileSystem delegate = getDelegate(path);
    return delegate.statNullable(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  @Nullable
  protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.statIfFound(adjustPath(path, delegate), followSymlinks);
  }

  /**
   * Retrieves the directory entries for the specified path under the assumption
   * that {@code resolvedPath} is the resolved path of {@code path} in one of the
   * underlying file systems.
   *
   * @param path the {@link Path} whose children are to be retrieved
   */
  @Override
  protected Collection<Path> getDirectoryEntries(Path path) throws IOException {
    FileSystem delegate = getDelegate(path);
    Path resolvedPath = adjustPath(path, delegate);
    Collection<Path> entries = resolvedPath.getDirectoryEntries();
    Collection<Path> result = Lists.newArrayListWithCapacity(entries.size());
    for (Path entry : entries) {
      result.add(path.getChild(entry.getBaseName()));
    }
    return result;
  }

  // No need for the more complex logic of getDirectoryEntries; it calls it implicitly.
  @Override
  protected Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.readdir(adjustPath(path, delegate), followSymlinks);
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.isReadable(adjustPath(path, delegate));
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    checkModifiable();
    FileSystem delegate = getDelegate(path);
    delegate.setReadable(adjustPath(path, delegate), readable);
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    if (!supportsModifications()) {
      return false;
    }
    FileSystem delegate = getDelegate(path);
    return delegate.isWritable(adjustPath(path, delegate));
  }

  @Override
  protected void setWritable(Path path, boolean writable) throws IOException {
    checkModifiable();
    FileSystem delegate = getDelegate(path);
    delegate.setWritable(adjustPath(path, delegate), writable);
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.isExecutable(adjustPath(path, delegate));
  }

  @Override
  protected void setExecutable(Path path, boolean executable) throws IOException {
    checkModifiable();
    FileSystem delegate = getDelegate(path);
    delegate.setExecutable(adjustPath(path, delegate), executable);
  }

  @Override
  protected String getFastDigestFunctionType(Path path) {
    FileSystem delegate = getDelegate(path);
    return delegate.getFastDigestFunctionType(adjustPath(path, delegate));
  }

  @Override
  protected byte[] getFastDigest(Path path) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.getFastDigest(adjustPath(path, delegate));
  }

  @Override
  protected byte[] getxattr(Path path, String name) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.getxattr(adjustPath(path, delegate), name);
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    FileSystem delegate = getDelegate(path);
    return delegate.getInputStream(adjustPath(path, delegate));
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append) throws IOException {
    checkModifiable();
    FileSystem delegate = getDelegate(path);
    return delegate.getOutputStream(adjustPath(path, delegate), append);
  }

  @Override
  protected void renameTo(Path sourcePath, Path targetPath) throws IOException {
    checkModifiable();
    FileSystem sourceDelegate = getDelegate(sourcePath);
    if (!sourceDelegate.supportsModifications()) {
      throw new UnsupportedOperationException(
          "The filesystem for the source path "
          + sourcePath.getPathString() + " does not support modifications.");
    }
    sourcePath = adjustPath(sourcePath, sourceDelegate);

    FileSystem targetDelegate = getDelegate(targetPath);
    if (!targetDelegate.supportsModifications()) {
      throw new UnsupportedOperationException(
          "The filesystem for the target path "
          + targetPath.getPathString() + " does not support modifications.");
    }
    targetPath = adjustPath(targetPath, targetDelegate);

    if (sourceDelegate == targetDelegate) {
      // Easy, same filesystem.
      sourceDelegate.renameTo(sourcePath, targetPath);
      return;
    } else {
      // Copy across filesystems, then delete.
      // copyFile throws on failure, so delete will never be reached if it fails.
      FileSystemUtils.copyFile(sourcePath, targetPath);
      sourceDelegate.delete(sourcePath);
    }
  }
}
