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
//
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;
import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.FileType;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A local file path representing a file on the host machine. You should use this when you want to
 * access local files via the file system.
 *
 * <p>Paths are always absolute.
 *
 * <p>Strings are normalized with '.' and '..' removed and resolved (if possible), any multiple
 * slashes ('/') removed, and any trailing slash also removed. Windows drive letters are uppercased.
 * The current implementation does not touch the incoming path string unless the string actually
 * needs to be normalized.
 *
 * <p>There is some limited support for Windows-style paths. Most importantly, drive identifiers in
 * front of a path (c:/abc) are supported and such paths are correctly recognized as absolute, as
 * are paths with backslash separators (C:\\foo\\bar). However, advanced Windows-style features like
 * \\\\network\\paths and \\\\?\\unc\\paths are not supported. We are currently using forward
 * slashes ('/') even on Windows, so backslashes '\' get converted to forward slashes during
 * normalization.
 *
 * <p>Mac and Windows file paths are case insensitive. Case is preserved.
 */
@ThreadSafe
@AutoCodec
public class Path implements Comparable<Path>, Serializable, FileType.HasFileType {
  private static FileSystem fileSystemForSerialization;

  /**
   * We need to specify used FileSystem. In this case we can save memory during the serialization.
   */
  public static void setFileSystemForSerialization(FileSystem fileSystem) {
    fileSystemForSerialization = fileSystem;
  }

  /**
   * Returns FileSystem that we are using.
   */
  public static FileSystem getFileSystemForSerialization() {
    return fileSystemForSerialization;
  }

  private static final OsPathPolicy OS = OsPathPolicy.getFilePathOs();
  private static final char SEPARATOR = OS.getSeparator();

  private String path;
  private int driveStrLength; // 1 on Unix, 3 on Windows
  private FileSystem fileSystem;

  /** Creates a local path that is specific to the host OS. */
  static Path create(String path, FileSystem fileSystem) {
    Preconditions.checkNotNull(path);
    int normalizationLevel = OS.needsToNormalize(path);
    String normalizedPath = OS.normalize(path, normalizationLevel);
    return createAlreadyNormalized(normalizedPath, fileSystem);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static Path createAlreadyNormalized(String path, FileSystem fileSystem) {
    int driveStrLength = OS.getDriveStrLength(path);
    return createAlreadyNormalized(path, driveStrLength, fileSystem);
  }

  static Path createAlreadyNormalized(String path, int driveStrLength, FileSystem fileSystem) {
    return new Path(path, driveStrLength, fileSystem);
  }

  /** This method expects path to already be normalized. */
  private Path(String path, int driveStrLength, FileSystem fileSystem) {
    Preconditions.checkArgument(driveStrLength > 0, "Paths must be absolute: '%s'", path);
    this.path = Preconditions.checkNotNull(path);
    this.driveStrLength = driveStrLength;
    this.fileSystem = fileSystem;
  }

  public String getPathString() {
    return path;
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return path;
  }

  /**
   * Returns the name of the leaf file or directory.
   *
   * <p>If called on a {@link Path} instance for a mount name (eg. '/' or 'C:/'), the empty string
   * is returned.
   */
  public String getBaseName() {
    int lastSeparator = path.lastIndexOf(SEPARATOR);
    return lastSeparator < driveStrLength
        ? path.substring(driveStrLength)
        : path.substring(lastSeparator + 1);
  }

  /** Synonymous with {@link Path#getRelative(String)}. */
  public Path getChild(String child) {
    FileSystemUtils.checkBaseName(child);
    return getRelative(child);
  }

  /**
   * Returns a {@link Path} instance representing the relative path between this {@link Path} and
   * the given path.
   */
  public Path getRelative(PathFragment other) {
    Preconditions.checkNotNull(other);
    String otherStr = other.getPathString();
    // Fast-path: The path fragment is already normal, use cheaper normalization check
    return getRelative(otherStr, other.getDriveStrLength(), OS.needsToNormalizeSuffix(otherStr));
  }

  /**
   * Returns a {@link Path} instance representing the relative path between this {@link Path} and
   * the given path.
   */
  public Path getRelative(String other) {
    Preconditions.checkNotNull(other);
    return getRelative(other, OS.getDriveStrLength(other), OS.needsToNormalize(other));
  }

  private Path getRelative(String other, int otherDriveStrLength, int normalizationLevel) {
    if (other.isEmpty()) {
      return this;
    }
    // This is an absolute path, simply return it
    if (otherDriveStrLength > 0) {
      String normalizedPath = OS.normalize(other, normalizationLevel);
      return new Path(normalizedPath, otherDriveStrLength, fileSystem);
    }
    String newPath;
    if (path.length() == driveStrLength) {
      newPath = path + other;
    } else {
      newPath = path + '/' + other;
    }
    // Note that even if other came from a PathFragment instance we still might
    // need to normalize the result if (for instance) other is a path that
    // starts with '..'
    newPath = OS.normalize(newPath, normalizationLevel);
    return new Path(newPath, driveStrLength, fileSystem);
  }

  /**
   * Returns the parent directory of this {@link Path}.
   *
   * <p>If called on a root (like '/'), it returns null.
   */
  @Nullable
  public Path getParentDirectory() {
    int lastSeparator = path.lastIndexOf(SEPARATOR);
    if (lastSeparator < driveStrLength) {
      if (path.length() > driveStrLength) {
        String newPath = path.substring(0, driveStrLength);
        return new Path(newPath, driveStrLength, fileSystem);
      } else {
        return null;
      }
    }
    String newPath = path.substring(0, lastSeparator);
    return new Path(newPath, driveStrLength, fileSystem);
  }

  /**
   * Returns the drive.
   *
   * <p>On unix, this will return "/". On Windows it will return the drive letter, like "C:/".
   */
  public String getDriveStr() {
    return path.substring(0, driveStrLength);
  }

  /**
   * Returns the {@link Path} relative to the base {@link Path}.
   *
   * <p>For example, <code>Path.create("foo/bar/wiz").relativeTo(Path.create("foo"))
   * </code> returns <code>Path.create("bar/wiz")</code>.
   *
   * <p>If the {@link Path} is not a child of the passed {@link Path} an {@link
   * IllegalArgumentException} is thrown. In particular, this will happen whenever the two {@link
   * Path} instances aren't both absolute or both relative.
   */
  public PathFragment relativeTo(Path base) {
    Preconditions.checkNotNull(base);
    checkSameFileSystem(base);
    String basePath = base.path;
    if (!OS.startsWith(path, basePath)) {
      throw new IllegalArgumentException(
          String.format("Path '%s' is not under '%s', cannot relativize", this, base));
    }
    int bn = basePath.length();
    if (bn == 0) {
      return PathFragment.createAlreadyNormalized(path, driveStrLength);
    }
    if (path.length() == bn) {
      return PathFragment.EMPTY_FRAGMENT;
    }
    final int lastSlashIndex;
    if (basePath.charAt(bn - 1) == '/') {
      lastSlashIndex = bn - 1;
    } else {
      lastSlashIndex = bn;
    }
    if (path.charAt(lastSlashIndex) != '/') {
      throw new IllegalArgumentException(
          String.format("Path '%s' is not under '%s', cannot relativize", this, base));
    }
    String newPath = path.substring(lastSlashIndex + 1);
    return PathFragment.createAlreadyNormalized(newPath, 0);
  }

  /**
   * Returns whether this path is an ancestor of another path.
   *
   * <p>A path is considered an ancestor of itself.
   */
  public boolean startsWith(Path other) {
    if (fileSystem != other.fileSystem) {
      return false;
    }
    return startsWith(other.path, other.driveStrLength);
  }

  /**
   * Returns whether this path is an ancestor of another path.
   *
   * <p>A path is considered an ancestor of itself.
   *
   * <p>An absolute path can never be an ancestor of a relative path fragment.
   */
  public boolean startsWith(PathFragment other) {
    if (!other.isAbsolute()) {
      return false;
    }
    String otherPath = other.getPathString();
    return startsWith(otherPath, OS.getDriveStrLength(otherPath));
  }

  private boolean startsWith(String otherPath, int otherDriveStrLength) {
    Preconditions.checkNotNull(otherPath);
    if (otherPath.length() > path.length()) {
      return false;
    }
    if (driveStrLength != otherDriveStrLength) {
      return false;
    }
    if (!OS.startsWith(path, otherPath)) {
      return false;
    }
    return path.length() == otherPath.length() // Handle equal paths
        || otherPath.length() == driveStrLength // Handle (eg.) 'C:/foo' starts with 'C:/'
        // Handle 'true' ancestors, eg. "foo/bar" starts with "foo", but does not start with "fo"
        || path.charAt(otherPath.length()) == SEPARATOR;
  }

  public FileSystem getFileSystem() {
    return fileSystem;
  }

  public PathFragment asFragment() {
    return PathFragment.createAlreadyNormalized(path, driveStrLength);
  }

  @Override
  public String toString() {
    return path;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Path other = (Path) o;
    if (fileSystem != other.fileSystem) {
      return false;
    }
    return OS.equals(this.path, other.path);
  }

  @Override
  public int hashCode() {
    // Do not include file system for efficiency.
    // In practice we never construct paths from different file systems.
    return OS.hash(this.path);
  }

  @Override
  public int compareTo(Path o) {
    // If they are on different file systems, the file system decides the ordering.
    FileSystem otherFs = o.getFileSystem();
    if (!fileSystem.equals(otherFs)) {
      int thisFileSystemHash = System.identityHashCode(fileSystem);
      int otherFileSystemHash = System.identityHashCode(otherFs);
      if (thisFileSystemHash < otherFileSystemHash) {
        return -1;
      } else if (thisFileSystemHash > otherFileSystemHash) {
        return 1;
      }
    }
    return OS.compare(this.path, o.path);
  }

  /** Returns true iff this path denotes an existing file of any kind. Follows symbolic links. */
  public boolean exists() {
    return fileSystem.exists(this, true);
  }

  /**
   * Returns true iff this path denotes an existing file of any kind.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a file other than a symbolic link is found
   */
  public boolean exists(Symlinks followSymlinks) {
    return fileSystem.exists(this, followSymlinks.toBoolean());
  }

  /**
   * Returns a new, immutable collection containing the names of all entities within the directory
   * denoted by the current path. Follows symbolic links.
   *
   * @throws FileNotFoundException If the directory is not found
   * @throws IOException If the path does not denote a directory
   */
  public Collection<Path> getDirectoryEntries() throws IOException, FileNotFoundException {
    Collection<String> entries = fileSystem.getDirectoryEntries(this);
    Collection<Path> result = new ArrayList<>(entries.size());
    for (String entry : entries) {
      result.add(getChild(entry));
    }
    return result;
  }

  /**
   * Returns a collection of the names and types of all entries within the directory denoted by the
   * current path. Follows symbolic links if {@code followSymlinks} is true. Note that the order of
   * the returned entries is not guaranteed.
   *
   * @param followSymlinks whether to follow symlinks or not
   * @throws FileNotFoundException If the directory is not found
   * @throws IOException If the path does not denote a directory
   */
  public Collection<Dirent> readdir(Symlinks followSymlinks) throws IOException {
    return fileSystem.readdir(this, followSymlinks.toBoolean());
  }

  /**
   * Returns the status of a file, following symbolic links.
   *
   * @throws IOException if there was an error obtaining the file status. Note, some implementations
   *     may defer the I/O, and hence the throwing of the exception, until the accessor methods of
   *     {@code FileStatus} are called.
   */
  public FileStatus stat() throws IOException {
    return fileSystem.stat(this, true);
  }

  /** Like stat(), but returns null on file-nonexistence instead of throwing. */
  public FileStatus statNullable() {
    return statNullable(Symlinks.FOLLOW);
  }

  /** Like stat(), but returns null on file-nonexistence instead of throwing. */
  public FileStatus statNullable(Symlinks symlinks) {
    return fileSystem.statNullable(this, symlinks.toBoolean());
  }

  /**
   * Returns the status of a file, optionally following symbolic links.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a file other than a symbolic link is found
   * @throws IOException if there was an error obtaining the file status. Note, some implementations
   *     may defer the I/O, and hence the throwing of the exception, until the accessor methods of
   *     {@code FileStatus} are called
   */
  public FileStatus stat(Symlinks followSymlinks) throws IOException {
    return fileSystem.stat(this, followSymlinks.toBoolean());
  }

  /**
   * Like {@link #stat}, but may return null if the file is not found (corresponding to {@code
   * ENOENT} and {@code ENOTDIR} in Unix's stat(2) function) instead of throwing. Follows symbolic
   * links.
   */
  public FileStatus statIfFound() throws IOException {
    return fileSystem.statIfFound(this, true);
  }

  /**
   * Like {@link #stat}, but may return null if the file is not found (corresponding to {@code
   * ENOENT} and {@code ENOTDIR} in Unix's stat(2) function) instead of throwing.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a file other than a symbolic link is found
   */
  public FileStatus statIfFound(Symlinks followSymlinks) throws IOException {
    return fileSystem.statIfFound(this, followSymlinks.toBoolean());
  }

  /** Returns true iff this path denotes an existing directory. Follows symbolic links. */
  public boolean isDirectory() {
    return fileSystem.isDirectory(this, true);
  }

  /**
   * Returns true iff this path denotes an existing directory.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a file other than a symbolic link is found
   */
  public boolean isDirectory(Symlinks followSymlinks) {
    return fileSystem.isDirectory(this, followSymlinks.toBoolean());
  }

  /**
   * Returns true iff this path denotes an existing regular or special file. Follows symbolic links.
   *
   * <p>For our purposes, "file" includes special files (socket, fifo, block or char devices) too;
   * it excludes symbolic links and directories.
   */
  public boolean isFile() {
    return fileSystem.isFile(this, true);
  }

  /**
   * Returns true iff this path denotes an existing regular or special file.
   *
   * <p>For our purposes, a "file" includes special files (socket, fifo, block or char devices) too;
   * it excludes symbolic links and directories.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a file other than a symbolic link is found.
   */
  public boolean isFile(Symlinks followSymlinks) {
    return fileSystem.isFile(this, followSymlinks.toBoolean());
  }

  /**
   * Returns true iff this path denotes an existing special file (e.g. fifo). Follows symbolic
   * links.
   */
  public boolean isSpecialFile() {
    return fileSystem.isSpecialFile(this, true);
  }

  /**
   * Returns true iff this path denotes an existing special file (e.g. fifo).
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a path other than a symbolic link is found.
   */
  public boolean isSpecialFile(Symlinks followSymlinks) {
    return fileSystem.isSpecialFile(this, followSymlinks.toBoolean());
  }

  /**
   * Returns true iff this path denotes an existing symbolic link. Does not follow symbolic links.
   */
  public boolean isSymbolicLink() {
    return fileSystem.isSymbolicLink(this);
  }

  /**
   * Returns an output stream to the file denoted by the current path, creating it and truncating it
   * if necessary. The stream is opened for writing.
   *
   * @throws FileNotFoundException If the file cannot be found or created.
   * @throws IOException If a different error occurs.
   */
  public OutputStream getOutputStream() throws IOException, FileNotFoundException {
    return getOutputStream(false);
  }

  /**
   * Returns an output stream to the file denoted by the current path, creating it and truncating it
   * if necessary. The stream is opened for writing.
   *
   * @param append whether to open the file in append mode.
   * @throws FileNotFoundException If the file cannot be found or created.
   * @throws IOException If a different error occurs.
   */
  public OutputStream getOutputStream(boolean append) throws IOException, FileNotFoundException {
    return fileSystem.getOutputStream(this, append);
  }

  /**
   * Creates a directory with the name of the current path, not following symbolic links. Returns
   * normally iff the directory exists after the call: true if the directory was created by this
   * call, false if the directory was already in existence. Throws an exception if the directory
   * could not be created for any reason.
   *
   * @throws IOException if the directory creation failed for any reason
   */
  public boolean createDirectory() throws IOException {
    return fileSystem.createDirectory(this);
  }

  /**
   * Ensures that the directory with the name of the current path and all its ancestor directories
   * exist.
   *
   * <p>Does not return whether the directory already existed or was created by some other
   * concurrent call to this method.
   *
   * @throws IOException if the directory creation failed for any reason
   */
  public void createDirectoryAndParents() throws IOException {
    fileSystem.createDirectoryAndParents(this);
  }

  /**
   * Creates a symbolic link with the name of the current path, following symbolic links. The
   * referent of the created symlink is is the absolute path "target"; it is not possible to create
   * relative symbolic links via this method.
   *
   * @throws IOException if the creation of the symbolic link was unsuccessful for any reason
   */
  public void createSymbolicLink(Path target) throws IOException {
    checkSameFileSystem(target);
    fileSystem.createSymbolicLink(this, target.asFragment());
  }

  /**
   * Creates a symbolic link with the name of the current path, following symbolic links. The
   * referent of the created symlink is is the path fragment "target", which may be absolute or
   * relative.
   *
   * @throws IOException if the creation of the symbolic link was unsuccessful for any reason
   */
  public void createSymbolicLink(PathFragment target) throws IOException {
    fileSystem.createSymbolicLink(this, target);
  }

  /**
   * Returns the target of the current path, which must be a symbolic link. The link contents are
   * returned exactly, and may contain an absolute or relative path. Analogous to readlink(2).
   *
   * <p>Note: for {@link FileSystem}s where {@link FileSystem#supportsSymbolicLinksNatively(Path)}
   * returns false, this method will throw an {@link UnsupportedOperationException} if the link
   * points to a non-existent file.
   *
   * @return the content (i.e. target) of the symbolic link
   * @throws IOException if the current path is not a symbolic link, or the contents of the link
   *     could not be read for any reason
   */
  public PathFragment readSymbolicLink() throws IOException {
    return fileSystem.readSymbolicLink(this);
  }

  /**
   * If the current path is a symbolic link, returns the target of this symbolic link. The semantics
   * are intentionally left underspecified otherwise to permit efficient implementations.
   *
   * @return the content (i.e. target) of the symbolic link
   * @throws IOException if the current path is not a symbolic link, or the contents of the link
   *     could not be read for any reason
   */
  public PathFragment readSymbolicLinkUnchecked() throws IOException {
    return fileSystem.readSymbolicLinkUnchecked(this);
  }

  /**
   * Create a hard link for the current path.
   *
   * @param link the path of the new link
   * @throws IOException if there was an error executing {@link FileSystem#createHardLink}
   */
  public void createHardLink(Path link) throws IOException {
    fileSystem.createHardLink(link, this);
  }

  /**
   * Returns the canonical path for this path, by repeatedly replacing symbolic links with their
   * referents. Analogous to realpath(3).
   *
   * @return the canonical path for this path
   * @throws IOException if any symbolic link could not be resolved, or other error occurred (for
   *     example, the path does not exist)
   */
  public Path resolveSymbolicLinks() throws IOException {
    return fileSystem.resolveSymbolicLinks(this);
  }

  /**
   * Renames the file denoted by the current path to the location "target", not following symbolic
   * links.
   *
   * <p>Files cannot be atomically renamed across devices; copying is required. Use {@link
   * FileSystemUtils#copyFile} followed by {@link Path#delete}.
   *
   * @throws IOException if the rename failed for any reason
   */
  public void renameTo(Path target) throws IOException {
    checkSameFileSystem(target);
    fileSystem.renameTo(this, target);
  }

  /**
   * Returns the size in bytes of the file denoted by the current path, following symbolic links.
   *
   * <p>The size of a directory or special file is undefined and should not be used.
   *
   * @throws FileNotFoundException if the file denoted by the current path does not exist
   * @throws IOException if the file's metadata could not be read, or some other error occurred
   */
  public long getFileSize() throws IOException, FileNotFoundException {
    return fileSystem.getFileSize(this, true);
  }

  /**
   * Returns the size in bytes of the file denoted by the current path.
   *
   * <p>The size of directory or special file is undefined. The size of a symbolic link is the
   * length of the name of its referent.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is deferenced until a file other than a symbol link is found
   * @throws FileNotFoundException if the file denoted by the current path does not exist
   * @throws IOException if the file's metadata could not be read, or some other error occurred
   */
  public long getFileSize(Symlinks followSymlinks) throws IOException, FileNotFoundException {
    return fileSystem.getFileSize(this, followSymlinks.toBoolean());
  }

  /**
   * Deletes the file denoted by this path, not following symbolic links. Returns normally iff the
   * file doesn't exist after the call: true if this call deleted the file, false if the file
   * already didn't exist. Throws an exception if the file could not be deleted but was present
   * prior to this call.
   *
   * @return true iff the file was actually deleted by this call
   * @throws IOException if the deletion failed but the file was present prior to the call
   */
  public boolean delete() throws IOException {
    return fileSystem.delete(this);
  }

  /**
   * Deletes all directory trees recursively beneath this path and removes the path as well.
   *
   * @throws IOException if the hierarchy cannot be removed successfully
   */
  public void deleteTree() throws IOException {
    fileSystem.deleteTree(this);
  }

  /**
   * Deletes all directory trees recursively beneath this path. Does nothing if the path is not a
   * directory.
   *
   * @throws IOException if the hierarchy cannot be removed successfully
   */
  public void deleteTreesBelow() throws IOException {
    fileSystem.deleteTreesBelow(this);
  }

  /**
   * Returns the last modification time of the file, in milliseconds since the UNIX epoch, of the
   * file denoted by the current path, following symbolic links.
   *
   * <p>Caveat: many filesystems store file times in seconds, so do not rely on the millisecond
   * precision.
   *
   * @throws IOException if the operation failed for any reason
   */
  public long getLastModifiedTime() throws IOException {
    return fileSystem.getLastModifiedTime(this, true);
  }

  /**
   * Returns the last modification time of the file, in milliseconds since the UNIX epoch, of the
   * file denoted by the current path.
   *
   * <p>Caveat: many filesystems store file times in seconds, so do not rely on the millisecond
   * precision.
   *
   * @param followSymlinks if {@link Symlinks#FOLLOW}, and this path denotes a symbolic link, the
   *     link is dereferenced until a file other than a symbolic link is found
   * @throws IOException if the modification time for the file could not be obtained for any reason
   */
  public long getLastModifiedTime(Symlinks followSymlinks) throws IOException {
    return fileSystem.getLastModifiedTime(this, followSymlinks.toBoolean());
  }

  /**
   * Sets the modification time of the file denoted by the current path. Follows symbolic links. If
   * newTime is -1, the current time according to the kernel is used; this may differ from the JVM's
   * clock.
   *
   * <p>Caveat: many filesystems store file times in seconds, so do not rely on the millisecond
   * precision.
   *
   * @param newTime time, in milliseconds since the UNIX epoch, or -1L, meaning use the kernel's
   *     current time
   * @throws IOException if the modification time for the file could not be set for any reason
   */
  public void setLastModifiedTime(long newTime) throws IOException {
    fileSystem.setLastModifiedTime(this, newTime);
  }

  /**
   * Returns the value of the given extended attribute name or null if the attribute does not exist
   * or the file system does not support extended attributes. Follows symlinks.
   */
  public byte[] getxattr(String name) throws IOException {
    return getxattr(name, Symlinks.FOLLOW);
  }

  /**
   * Returns the value of the given extended attribute name or null if the attribute does not exist
   * or the file system does not support extended attributes.
   *
   * @param followSymlinks whether to follow symlinks or not
   */
  public byte[] getxattr(String name, Symlinks followSymlinks) throws IOException {
    return fileSystem.getxattr(this, name, followSymlinks.toBoolean());
  }

  /**
   * Gets a fast digest for the given path, or {@code null} if there isn't one available. The digest
   * should be suitable for detecting changes to the file.
   */
  public byte[] getFastDigest() throws IOException {
    return fileSystem.getFastDigest(this);
  }

  /**
   * Returns the digest of the file denoted by the current path, following symbolic links.
   *
   * @return a new byte array containing the file's digest
   * @throws IOException if the digest could not be computed for any reason
   */
  public byte[] getDigest() throws IOException {
    return fileSystem.getDigest(this);
  }

  /**
   * Return a string representation, as hexadecimal digits, of some hash of the directory.
   *
   * <p>The hash itself is computed according to the design document
   * https://github.com/bazelbuild/proposals/blob/master/designs/2018-07-13-repository-hashing.md
   * and takes enough information into account, to detect the typical non-reproducibility
   * of source-like repository rules, while leaving out what will change from invocation to
   * invocation of a repository rule (in particular file owners) and can reasonably be ignored
   * when considering if a repository is "the same source tree".
   *
   * @return a string representation of the bash of the directory
   * @throws IOException if the digest could not be computed for any reason
   */
  public String getDirectoryDigest() throws IOException {
    List<String> entries = new ArrayList<String>(fileSystem.getDirectoryEntries(this));
    Collections.sort(entries);
    Hasher hasher = fileSystem.getDigestFunction().getHashFunction().newHasher();
    for (String entry : entries) {
      Path path = this.getChild(entry);
      FileStatus stat = path.stat(Symlinks.NOFOLLOW);
      hasher.putUnencodedChars(entry);
      if (stat.isFile()) {
        if (path.isExecutable()) {
          hasher.putChar('x');
        } else {
          hasher.putChar('-');
        }
        hasher.putBytes(path.getDigest());
      } else if (stat.isDirectory()) {
        hasher.putChar('d').putUnencodedChars(path.getDirectoryDigest());
      } else if (stat.isSymbolicLink()) {
        PathFragment link = path.readSymbolicLink();
        if (link.isAbsolute()) {
          try {
            Path resolved = path.resolveSymbolicLinks();
            if (resolved.isFile()) {
              if (resolved.isExecutable()) {
                hasher.putChar('x');
              } else {
                hasher.putChar('-');
              }
              hasher.putBytes(resolved.getDigest());
            } else {
              // link to a non-file: include the link itself in the hash
              hasher.putChar('l').putUnencodedChars(link.toString());
            }
          } catch (IOException e) {
            // dangling link: include the link itself in the hash
            hasher.putChar('l').putUnencodedChars(link.toString());
          }
        } else {
          // relative link: include the link itself in the hash
          hasher.putChar('l').putUnencodedChars(link.toString());
        }
      } else {
        // Neither file, nor directory, nor symlink. So do not include further information
        // in the hash, asuming it will not be used during the BUILD anyway.
        hasher.putChar('s');
      }
    }
    return hasher.hash().toString();
  }

  /**
   * Opens the file denoted by this path, following symbolic links, for reading, and returns an
   * input stream to it.
   *
   * @throws IOException if the file was not found or could not be opened for reading
   */
  public InputStream getInputStream() throws IOException {
    return fileSystem.getInputStream(this);
  }

  /**
   * Opens the file denoted by this path, following symbolic links, for reading, and returns a file
   * channel for it.
   *
   * @throws IOException if the file was not found or could not be opened for reading
   */
  public ReadableByteChannel createReadableByteChannel() throws IOException {
    return fileSystem.createReadableByteChannel(this);
  }

  /**
   * Returns a java.io.File representation of this path.
   *
   * <p>Caveat: the result may be useless if this path's getFileSystem() is not
   * the UNIX filesystem.
   */
  public File getPathFile() {
    return new File(getPathString());
  }

  /**
   * Returns true if the file denoted by the current path, following symbolic links, is writable for
   * the current user.
   *
   * @throws FileNotFoundException if the file does not exist, a dangling symbolic link was
   *     encountered, or the file's metadata could not be read
   */
  public boolean isWritable() throws IOException, FileNotFoundException {
    return fileSystem.isWritable(this);
  }

  /**
   * Sets the read permissions of the file denoted by the current path, following symbolic links.
   * Permissions apply to the current user.
   *
   * @param readable if true, the file is set to readable; otherwise the file is made non-readable
   * @throws FileNotFoundException if the file does not exist
   * @throws IOException If the action cannot be taken (ie. permissions)
   */
  public void setReadable(boolean readable) throws IOException, FileNotFoundException {
    fileSystem.setReadable(this, readable);
  }

  /**
   * Sets the write permissions of the file denoted by the current path, following symbolic links.
   * Permissions apply to the current user.
   *
   * <p>TODO(bazel-team): (2009) what about owner/group/others?
   *
   * @param writable if true, the file is set to writable; otherwise the file is made non-writable
   * @throws FileNotFoundException if the file does not exist
   * @throws IOException If the action cannot be taken (ie. permissions)
   */
  public void setWritable(boolean writable) throws IOException, FileNotFoundException {
    fileSystem.setWritable(this, writable);
  }

  /**
   * Returns true iff the file specified by the current path, following symbolic links, is
   * executable by the current user.
   *
   * @throws FileNotFoundException if the file does not exist or a dangling symbolic link was
   *     encountered
   * @throws IOException if some other I/O error occurred
   */
  public boolean isExecutable() throws IOException, FileNotFoundException {
    return fileSystem.isExecutable(this);
  }

  /**
   * Returns true iff the file specified by the current path, following symbolic links, is readable
   * by the current user.
   *
   * @throws FileNotFoundException if the file does not exist or a dangling symbolic link was
   *     encountered
   * @throws IOException if some other I/O error occurred
   */
  public boolean isReadable() throws IOException, FileNotFoundException {
    return fileSystem.isReadable(this);
  }

  /**
   * Sets the execute permission on the file specified by the current path, following symbolic
   * links. Permissions apply to the current user.
   *
   * @throws FileNotFoundException if the file does not exist or a dangling symbolic link was
   *     encountered
   * @throws IOException if the metadata change failed, for example because of permissions
   */
  public void setExecutable(boolean executable) throws IOException, FileNotFoundException {
    fileSystem.setExecutable(this, executable);
  }

  /**
   * Sets the permissions on the file specified by the current path, following symbolic links. If
   * permission changes on this path's {@link FileSystem} are slow (e.g. one syscall per change),
   * this method should aim to be faster than setting each permission individually. If this path's
   * {@link FileSystem} does not support group and others permissions, those bits will be ignored.
   *
   * @throws FileNotFoundException if the file does not exist or a dangling symbolic link was
   *     encountered
   * @throws IOException if the metadata change failed, for example because of permissions
   */
  public void chmod(int mode) throws IOException {
    fileSystem.chmod(this, mode);
  }

  public void prefetchPackageAsync(int maxDirs) {
    fileSystem.prefetchPackageAsync(this, maxDirs);
  }

  private void checkSameFileSystem(Path that) {
    if (this.fileSystem != that.fileSystem) {
      throw new IllegalArgumentException(
          "Files are on different filesystems: " + this + ", " + that);
    }
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    Preconditions.checkState(
        fileSystem == fileSystemForSerialization, "%s %s", fileSystem, fileSystemForSerialization);
    out.writeUTF(path);
  }

  private void readObject(ObjectInputStream in) throws IOException {
    path = in.readUTF();
    fileSystem = fileSystemForSerialization;
    driveStrLength = OS.getDriveStrLength(path);
  }
}
