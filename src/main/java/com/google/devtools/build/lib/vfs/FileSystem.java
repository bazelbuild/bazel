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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.ByteSource;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.FileAlreadyExistsException;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/** This interface models a file system. */
@ThreadSafe
public abstract class FileSystem {

  private final DigestHashFunction digestFunction;

  public FileSystem(DigestHashFunction digestFunction) {
    this.digestFunction = Preconditions.checkNotNull(digestFunction);
  }

  public DigestHashFunction getDigestFunction() {
    return digestFunction;
  }

  /**
   * An exception thrown when attempting to resolve an ordinary file as a symlink.
   */
  protected static final class NotASymlinkException extends IOException {
    public NotASymlinkException(Path path) {
      super(path + " is not a symlink");
    }
  }

  private final Root absoluteRoot = new Root.AbsoluteRoot(this);

  /**
   * Returns an absolute path instance, given an absolute path name, without double slashes, .., or
   * . segments. While this method will normalize the path representation by creating a
   * structured/parsed representation, it will not cause any IO. (e.g., it will not resolve symbolic
   * links if it's a Unix file system.
   */
  public Path getPath(String path) {
    return Path.create(path, this);
  }

  /** Returns an absolute path instance, given an absolute path fragment. */
  public Path getPath(PathFragment pathFragment) {
    Preconditions.checkArgument(pathFragment.isAbsolute(), "Not absolute: %s", pathFragment);
    return Path.createAlreadyNormalized(
        pathFragment.getPathString(), pathFragment.getDriveStrLength(), this);
  }

  final Root getAbsoluteRoot() {
    return absoluteRoot;
  }

  /**
   * Returns whether or not the FileSystem supports modifications of files and file entries.
   *
   * <p>Returns true if FileSystem supports the following:
   *
   * <ul>
   *   <li>{@link #setWritable(Path, boolean)}
   *   <li>{@link #setExecutable(Path, boolean)}
   * </ul>
   *
   * The above calls will result in an {@link UnsupportedOperationException} on a FileSystem where
   * this method returns {@code false}.
   */
  public abstract boolean supportsModifications(Path path);

  /**
   * Returns whether or not the FileSystem supports symbolic links.
   *
   * <p>Returns true if FileSystem supports the following:
   *
   * <ul>
   *   <li>{@link #createSymbolicLink(Path, PathFragment)}
   *   <li>{@link #getFileSize(Path, boolean)} where {@code followSymlinks=false}
   *   <li>{@link #getLastModifiedTime(Path, boolean)} where {@code followSymlinks=false}
   *   <li>{@link #readSymbolicLink(Path)} where the link points to a non-existent file
   * </ul>
   *
   * The above calls may result in an {@link UnsupportedOperationException} on a FileSystem where
   * this method returns {@code false}. The implementation can try to emulate these calls at its own
   * discretion.
   */
  public abstract boolean supportsSymbolicLinksNatively(Path path);

  /**
   * Returns whether or not the FileSystem supports hard links.
   *
   * <p>Returns true if FileSystem supports the following:
   *
   * <ul>
   *   <li>{@link #createFSDependentHardLink(Path, Path)}
   * </ul>
   *
   * The above calls may result in an {@link UnsupportedOperationException} on a FileSystem where
   * this method returns {@code false}. The implementation can try to emulate these calls at its own
   * discretion.
   */
  protected abstract boolean supportsHardLinksNatively(Path path);

  /***
   * Returns true if file path is case-sensitive on this file system. Default is true.
   */
  public abstract boolean isFilePathCaseSensitive();

  /**
   * Returns the type of the file system path belongs to.
   *
   * <p>The string returned is obtained directly from the operating system, so
   * it's a best guess in absence of a guaranteed api.
   *
   * <p>This implementation uses <code>/proc/mounts</code> to determine the
   * file system type.
   */
  public String getFileSystemType(Path path) {
    String fileSystem = "unknown";
    int bestMountPointSegmentCount = -1;
    try {
      Path canonicalPath = path.resolveSymbolicLinks();
      Path mountTable = path.getRelative("/proc/mounts");
      try (InputStreamReader reader = new InputStreamReader(mountTable.getInputStream(),
          ISO_8859_1)) {
        for (String line : CharStreams.readLines(reader)) {
          String[] words = line.split("\\s+");
          if (words.length >= 3) {
            if (!words[1].startsWith("/")) {
              continue;
            }
            Path mountPoint = path.getFileSystem().getPath(words[1]);
            int segmentCount = mountPoint.asFragment().segmentCount();
            if (canonicalPath.startsWith(mountPoint) && segmentCount > bestMountPointSegmentCount) {
              bestMountPointSegmentCount = segmentCount;
              fileSystem = words[2];
            }
          }
        }
      }
    } catch (IOException e) {
      // pass
    }
    return fileSystem;
  }

  /**
   * Creates a directory with the name of the current path. See {@link Path#createDirectory} for
   * specification.
   */
  public abstract boolean createDirectory(Path path) throws IOException;

  /**
   * Creates all directories up to the path. See {@link Path#createDirectoryAndParents} for
   * specification.
   */
  public abstract void createDirectoryAndParents(Path path) throws IOException;

  /**
   * Returns the size in bytes of the file denoted by {@code path}. See {@link
   * Path#getFileSize(Symlinks)} for specification.
   *
   * <p>Note: for <@link FileSystem>s where {@link #supportsSymbolicLinksNatively(Path)} returns
   * false, this method will throw an {@link UnsupportedOperationException} if {@code
   * followSymLinks=false}.
   */
  protected abstract long getFileSize(Path path, boolean followSymlinks) throws IOException;

  /** Deletes the file denoted by {@code path}. See {@link Path#delete} for specification. */
  public abstract boolean delete(Path path) throws IOException;

  /**
   * Deletes all directory trees recursively beneath the given path and removes that path as well.
   *
   * @param path the directory hierarchy to remove
   * @throws IOException if the hierarchy cannot be removed successfully
   */
  public void deleteTree(Path path) throws IOException {
    deleteTreesBelow(path);
    path.delete();
  }

  /**
   * Deletes all directory trees recursively beneath the given path. Does nothing if the given path
   * is not a directory.
   *
   * <p>This generic implementation is not as efficient as it could be: for example, we issue
   * separate stats for each directory entry to determine if they are directories or not (instead of
   * reusing the information that readdir returns), and we issue separate operations to toggle
   * different permissions while they could be done at once via chmod. Subclasses can optimize this
   * by taking advantage of platform-specific features.
   *
   * @param dir the directory hierarchy to remove
   * @throws IOException if the hierarchy cannot be removed successfully
   */
  public void deleteTreesBelow(Path dir) throws IOException {
    if (dir.isDirectory(Symlinks.NOFOLLOW)) {
      Collection<Path> entries;
      try {
        entries = dir.getDirectoryEntries();
      } catch (IOException e) {
        // If we couldn't read the directory, it may be because it's not readable. Try granting this
        // permission and retry. If the retry fails, give up.
        dir.setReadable(true);
        dir.setExecutable(true);
        entries = dir.getDirectoryEntries();
      }

      Iterator<Path> iterator = entries.iterator();
      if (iterator.hasNext()) {
        Path first = iterator.next();
        deleteTreesBelow(first);
        try {
          // If the directory is not executable, delete(), depending on implementation, may decide
          // that the directory entry does not exist and return false without throwing.
          if (!first.delete()) {
            throw new IOException(
                "Unable to delete \"" + first + "\": directory entry does not exist");
          }
        } catch (IOException e) {
          // If we couldn't delete the first entry in a directory, it may be because the directory
          // (not the entry!) is not writable or executable. Try granting this permission and retry.
          // If the retry fails, give up. Note that we have to retry deleteTreesBelow() too in case
          // first is itself a directory; if the directory were not executable, the initial
          // first.deleteTreesBelow() call would have been a silent no-op (since first.isDirectory()
          // would have returned false) and sub-entries of first would not have been deleted.
          dir.setWritable(true);
          dir.setExecutable(true);
          deleteTreesBelow(first);
          first.delete();
        }
      }
      while (iterator.hasNext()) {
        Path path = iterator.next();
        deleteTreesBelow(path);
        // No need to retry here: if needed, we already unprotected the directory earlier.
        path.delete();
      }
    }
  }

  /**
   * Returns the last modification time of the file denoted by {@code path}. See {@link
   * Path#getLastModifiedTime(Symlinks)} for specification.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsSymbolicLinksNatively(Path)} returns
   * false, this method will throw an {@link UnsupportedOperationException} if {@code
   * followSymLinks=false}.
   */
  protected abstract long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException;

  /**
   * Sets the last modification time of the file denoted by {@code path}. See {@link
   * Path#setLastModifiedTime} for specification.
   */
  public abstract void setLastModifiedTime(Path path, long newTime) throws IOException;

  /**
   * Returns value of the given extended attribute name or null if attribute does not exist or file
   * system does not support extended attributes.
   *
   * <p>Default implementation assumes that file system does not support extended attributes and
   * always returns null. Specific file system implementations should override this method if they
   * do provide support for extended attributes.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @param followSymlinks whether to follow symlinks or not; if false, returns the xattr of the
   *     link itself, not its target.
   * @return the value of the extended attribute associated with 'path', if any, or null if no such
   *     attribute is defined (ENODATA) or file system does not support extended attributes at all.
   * @throws IOException if the call failed for any other reason.
   */
  public byte[] getxattr(Path path, String name, boolean followSymlinks) throws IOException {
    return null;
  }

  /**
   * Gets a fast digest for the given path, or {@code null} if there isn't one available or the
   * filesystem doesn't support them. This digest should be suitable for detecting changes to the
   * file.
   */
  protected byte[] getFastDigest(Path path) throws IOException {
    return null;
  }

  /**
   * Returns the digest of the file denoted by the path, following symbolic links.
   *
   * <p>Subclasses may (and do) optimize this computation for a particular digest functions.
   *
   * @return a new byte array containing the file's digest
   * @throws IOException if the digest could not be computed for any reason
   */
  protected byte[] getDigest(final Path path) throws IOException {
    return new ByteSource() {
      @Override
      public InputStream openStream() throws IOException {
        return getInputStream(path);
      }
    }.hash(digestFunction.getHashFunction()).asBytes();
  }

  /**
   * Returns true if "path" denotes an existing symbolic link. See
   * {@link Path#isSymbolicLink} for specification.
   */
  protected abstract boolean isSymbolicLink(Path path);

  /**
   * Appends a single regular path segment 'child' to 'dir', recursively
   * resolving symbolic links in 'child'. 'dir' must be canonical. 'maxLinks' is
   * the maximum number of symbolic links that may be traversed before it gives
   * up (the Linux kernel uses 32).
   *
   * <p>(This method does not need to be synchronized; but the result may be
   * stale in the case of concurrent modification.)
   *
   * @throws IOException if 'dir' is not an existing directory; or if
   *         stat(child) fails for any reason, or if 'child' is a symlink and
   *         readlink(child) fails for any reason (e.g. ENOENT, EACCES), or if
   *         the chain of symbolic links exceeds 'maxLinks'.
   */
  protected final Path appendSegment(Path dir, String child, int maxLinks) throws IOException {
    Path naive = dir.getChild(child);

    PathFragment linkTarget = resolveOneLink(naive);
    if (linkTarget == null) {
      return naive; // regular file or directory
    }

    if (maxLinks-- == 0) {
      throw new IOException(naive + " (Too many levels of symbolic links)");
    }
    if (linkTarget.isAbsolute()) {
      dir = getPath(linkTarget.getDriveStr());
    }
    for (String name : linkTarget.segments()) {
      if (name.equals(".") || name.isEmpty()) {
        // no-op
      } else if (name.equals("..")) {
        Path parent = dir.getParentDirectory();
        // root's parent is root, when canonicalizing, so this is a no-op.
        if (parent != null) { dir = parent; }
      } else {
        dir = appendSegment(dir, name, maxLinks);
      }
    }
    return dir;
  }

  /**
   * Helper method of {@link #resolveSymbolicLinks(Path)}. This method
   * encapsulates the I/O component of a full canonicalization operation.
   * Subclasses can (and do) provide more efficient implementations.
   *
   * <p>(This method does not need to be synchronized; but the result may be
   * stale in the case of concurrent modification.)
   *
   * @param path a path, of which all but the last segment is guaranteed to be
   *        canonical
   * @return {@link #readSymbolicLink} iff path is a symlink or null iff
   *         path exists but is not a symlink
   * @throws IOException if the file did not exist, or a parent directory could
   *         not be searched
   */
  protected PathFragment resolveOneLink(Path path) throws IOException {
    try {
      return readSymbolicLink(path);
    } catch (NotASymlinkException e) {
      // Not a symbolic link.  Check it exists.

      // (A simple call to lstat would replace all of this.)
      if (!exists(path, false)) {
        throw new FileNotFoundException(path + " (No such file or directory)");
      }

      // TODO(bazel-team): (2009) ideally, throw ENOTDIR if dir is not a dir, but that
      // would require twice as many stats, or a much more convoluted
      // implementation (like glibc's canonicalize.c).

      return null; //  exists.
    }
  }

  /**
   * Returns the canonical path for the given path. See
   * {@link Path#resolveSymbolicLinks} for specification.
   */
  protected Path resolveSymbolicLinks(Path path)
      throws IOException {
    Path parentNode = path.getParentDirectory();
    return parentNode == null
        ? path // (root)
        : appendSegment(resolveSymbolicLinks(parentNode), path.getBaseName(), 32);
  }

  /**
   * Returns the status of a file. See {@link Path#stat(Symlinks)} for
   * specification.
   *
   * <p>The default implementation of this method is a "lazy" one, based on
   * other accessor methods such as {@link #isFile}, etc. Subclasses may provide
   * more efficient specializations. However, we still try to follow Unix-like
   * semantics of failing fast in case of non-existent files (or in case of
   * permission issues).
   */
  protected FileStatus stat(final Path path, final boolean followSymlinks) throws IOException {
    FileStatus status = new FileStatus() {
      volatile Boolean isFile;
      volatile Boolean isDirectory;
      volatile Boolean isSymbolicLink;
      volatile Boolean isSpecial;
      volatile long size = -1;
      volatile long mtime = -1;

      @Override
      public boolean isFile() {
        if (isFile == null) { isFile = FileSystem.this.isFile(path, followSymlinks); }
        return isFile;
      }

      @Override
      public boolean isDirectory() {
        if (isDirectory == null) {
          isDirectory = FileSystem.this.isDirectory(path, followSymlinks);
        }
        return isDirectory;
      }

      @Override
      public boolean isSymbolicLink() {
        if (isSymbolicLink == null)  { isSymbolicLink = FileSystem.this.isSymbolicLink(path); }
        return isSymbolicLink;
      }

      @Override
      public boolean isSpecialFile() {
        if (isSpecial == null)  { isSpecial = FileSystem.this.isSpecialFile(path, followSymlinks); }
        return isSpecial;
      }

      @Override
      public long getSize() throws IOException {
        if (size == -1) { size = getFileSize(path, followSymlinks); }
        return size;
      }

      @Override
      public long getLastModifiedTime() throws IOException {
        if (mtime == -1) { mtime = FileSystem.this.getLastModifiedTime(path, followSymlinks); }
        return mtime;
      }

      @Override
      public long getLastChangeTime() {
        throw new UnsupportedOperationException();
      }

      @Override
      public long getNodeId() {
        throw new UnsupportedOperationException();
      }
    };

    // Fail fast in case if some operations will actually fail, since stat() call sometimes used
    // to verify file existence as well. We will use getLastModifiedTime() method for that purpose.
    status.getLastModifiedTime();

    return status;
  }

  /**
   * Like stat(), but returns null on failures instead of throwing.
   */
  protected FileStatus statNullable(Path path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks);
    } catch (IOException e) {
      return null;
    }
  }

  /**
   * Like {@link #stat}, but returns null if the file is not found (corresponding to
   * {@code ENOENT} or {@code ENOTDIR} in Unix's stat(2) function) instead of throwing. Note that
   * this implementation does <i>not</i> successfully catch {@code ENOTDIR} exceptions. If the
   * instantiated filesystem can catch such errors, it should override this method to do so.
   */
  protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
    try {
      return stat(path, followSymlinks);
    } catch (FileNotFoundException e) {
      return null;
    }
  }

  /**
   * Returns true iff {@code path} denotes an existing directory. See
   * {@link Path#isDirectory(Symlinks)} for specification.
   */
  protected abstract boolean isDirectory(Path path, boolean followSymlinks);

  /**
   * Returns true iff {@code path} denotes an existing regular or special file.
   * See {@link Path#isFile(Symlinks)} for specification.
   */
  protected abstract boolean isFile(Path path, boolean followSymlinks);

  /**
   * Returns true iff {@code path} denotes a special file.
   * See {@link Path#isSpecialFile(Symlinks)} for specification.
   */
  protected abstract boolean isSpecialFile(Path path, boolean followSymlinks);

  /**
   * Creates a symbolic link. See {@link Path#createSymbolicLink(Path)} for specification.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsSymbolicLinksNatively(Path)} returns
   * false, this method will throw an {@link UnsupportedOperationException}
   */
  protected abstract void createSymbolicLink(Path linkPath, PathFragment targetFragment)
      throws IOException;

  /**
   * Returns the target of a symbolic link. See {@link Path#readSymbolicLink} for specification.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsSymbolicLinksNatively(Path)} returns
   * false, this method will throw an {@link UnsupportedOperationException} if the link points to a
   * non-existent file.
   *
   * @throws NotASymlinkException if the current path is not a symbolic link
   * @throws IOException if the contents of the link could not be read for any reason.
   */
  protected abstract PathFragment readSymbolicLink(Path path) throws IOException;

  /**
   * Returns the target of a symbolic link, under the assumption that the given path is indeed a
   * symbolic link (this assumption permits efficient implementations). See
   * {@link Path#readSymbolicLinkUnchecked} for specification.
   *
   * @throws IOException if the contents of the link could not be read for any reason.
   */
  protected PathFragment readSymbolicLinkUnchecked(Path path) throws IOException {
    return readSymbolicLink(path);
  }

  /** Returns true iff this path denotes an existing file of any kind. Follows symbolic links. */
  public boolean exists(Path path) {
    return exists(path, true);
  }

  /**
   * Returns true iff {@code path} denotes an existing file of any kind. See
   * {@link Path#exists(Symlinks)} for specification.
   */
  protected abstract boolean exists(Path path, boolean followSymlinks);

  /**
   * Returns a collection containing the names of all entities within the directory denoted by the
   * {@code path}.
   *
   * @throws IOException if there was an error reading the directory entries
   */
  protected abstract Collection<String> getDirectoryEntries(Path path) throws IOException;

  protected static Dirent.Type direntFromStat(FileStatus stat) {
    if (stat == null) {
      return Dirent.Type.UNKNOWN;
    } else if (stat.isSpecialFile()) {
      return Dirent.Type.UNKNOWN;
    } else if (stat.isFile()) {
      return Dirent.Type.FILE;
    } else if (stat.isDirectory()) {
      return Dirent.Type.DIRECTORY;
    } else if (stat.isSymbolicLink()) {
      return Dirent.Type.SYMLINK;
    } else {
      return Dirent.Type.UNKNOWN;
    }
  }

  /**
   * Returns a Dirents structure, listing the names of all entries within the
   * directory {@code path}, plus their types (file, directory, other).
   *
   * @param followSymlinks whether to follow symlinks when determining the file types of
   *     individual directory entries. No matter the value of this parameter, symlinks are
   *     followed when resolving the directory whose entries are to be read.
   * @throws IOException if there was an error reading the directory entries
   */
  protected Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
    Collection<String> children = getDirectoryEntries(path);
    List<Dirent> dirents = Lists.newArrayListWithCapacity(children.size());
    for (String child : children) {
      Path childPath = path.getChild(child);
      Dirent.Type type = direntFromStat(statNullable(childPath, followSymlinks));
      dirents.add(new Dirent(child, type));
    }
    return dirents;
  }

  /**
   * Returns true iff the file represented by {@code path} is readable.
   *
   * @throws IOException if there was an error reading the file's metadata
   */
  protected abstract boolean isReadable(Path path) throws IOException;

  /**
   * Sets the file to readable (if the argument is true) or non-readable (if the argument is false)
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(Path)} returns false or
   * which do not support unreadable files, this method will throw an {@link
   * UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  protected abstract void setReadable(Path path, boolean readable) throws IOException;

  /**
   * Returns true iff the file represented by {@code path} is writable.
   *
   * @throws IOException if there was an error reading the file's metadata
   */
  protected abstract boolean isWritable(Path path) throws IOException;

  /**
   * Sets the file to writable (if the argument is true) or non-writable (if the argument is false)
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(Path)} returns false, this
   * method will throw an {@link UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  public abstract void setWritable(Path path, boolean writable) throws IOException;

  /**
   * Returns true iff the file represented by the path is executable.
   *
   * @throws IOException if there was an error reading the file's metadata
   */
  protected abstract boolean isExecutable(Path path) throws IOException;

  /**
   * Sets the file to executable, if the argument is true. It is currently not supported to unset
   * the executable status of a file, so {code executable=false} yields an {@link
   * UnsupportedOperationException}.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(Path)} returns false, this
   * method will throw an {@link UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  protected abstract void setExecutable(Path path, boolean executable) throws IOException;

  /**
   * Sets the file permissions. If permission changes on this {@link FileSystem} are slow (e.g. one
   * syscall per change), this method should aim to be faster than setting each permission
   * individually. If this {@link FileSystem} does not support group or others permissions, those
   * bits will be ignored.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(Path)} returns false, this
   * method will throw an {@link UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  protected void chmod(Path path, int mode) throws IOException {
    setReadable(path, (mode & 0400) != 0);
    setWritable(path, (mode & 0200) != 0);
    setExecutable(path, (mode & 0100) != 0);
  }

  /**
   * Creates an InputStream accessing the file denoted by the path.
   *
   * @throws IOException if there was an error opening the file for reading
   */
  protected abstract InputStream getInputStream(Path path) throws IOException;

  /**
   * Creates a ReadableFileChannel accessing the file denoted by the path.
   *
   * @throws IOException if there was an error opening the file for reading
   */
  protected ReadableByteChannel createReadableByteChannel(Path path) throws IOException {
    throw new UnsupportedOperationException();
  }

  /**
   * Creates an OutputStream accessing the file denoted by path.
   *
   * @throws IOException if there was an error opening the file for writing
   */
  protected final OutputStream getOutputStream(Path path) throws IOException {
    return getOutputStream(path, false);
  }

  /**
   * Creates an OutputStream accessing the file denoted by path.
   *
   * @param append whether to open the output stream in append mode
   * @throws IOException if there was an error opening the file for writing
   */
  protected abstract OutputStream getOutputStream(Path path, boolean append) throws IOException;

  /**
   * Renames the file denoted by "sourceNode" to the location "targetNode". See {@link
   * Path#renameTo} for specification.
   */
  public abstract void renameTo(Path sourcePath, Path targetPath) throws IOException;

  /**
   * Create a new hard link file at "linkPath" for file at "originalPath".
   *
   * @param linkPath The path of the new link file to be created
   * @param originalPath The path of the original file
   * @throws IOException if the original file does not exist or the link file already exists
   */
  protected void createHardLink(Path linkPath, Path originalPath) throws IOException {

    if (!originalPath.exists()) {
      throw new FileNotFoundException(
          "File \""
              + originalPath.getBaseName()
              + "\" linked from \""
              + linkPath.getBaseName()
              + "\" does not exist");
    }

    if (linkPath.exists()) {
      throw new FileAlreadyExistsException(
          "New link file \"" + linkPath.getBaseName() + "\" already exists");
    }

    createFSDependentHardLink(linkPath, originalPath);
  }

  /**
   * Create a new hard link file at "linkPath" for file at "originalPath".
   *
   * @param linkPath The path of the new link file to be created
   * @param originalPath The path of the original file
   * @throws IOException if there was an I/O error
   */
  protected abstract void createFSDependentHardLink(Path linkPath, Path originalPath)
      throws IOException;

  /**
   * Prefetch all directories and symlinks within the package
   * rooted at "path".  Enter at most "maxDirs" total directories.
   * Specializations for high-latency remote filesystems may wish to
   * implement this in order to warm the filesystem's internal caches.
   */
  protected void prefetchPackageAsync(Path path, int maxDirs) { }
}
