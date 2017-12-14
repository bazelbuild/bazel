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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteSource;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import com.google.devtools.common.options.EnumConverter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.FileAlreadyExistsException;
import java.util.Collection;
import java.util.List;

/**
 * This interface models a file system using UNIX the naming scheme.
 */
@ThreadSafe
public abstract class FileSystem {
  /** Type of hash function to use for digesting files. */
  // The underlying HashFunctions are immutable and thread safe.
  @SuppressWarnings("ImmutableEnumChecker")
  public enum HashFunction {
    MD5(Hashing.md5()),
    SHA1(Hashing.sha1()),
    SHA256(Hashing.sha256());

    private final com.google.common.hash.HashFunction hash;

    HashFunction(com.google.common.hash.HashFunction hash) {
      this.hash = hash;
    }

    /** Converts to {@link HashFunction}. */
    public static class Converter extends EnumConverter<HashFunction> {
      public Converter() {
        super(HashFunction.class, "hash function");
      }
    }

    public com.google.common.hash.HashFunction getHash() {
      return hash;
    }

    public boolean isValidDigest(byte[] digest) {
      return digest != null && digest.length * 8 == hash.bits();
    }
  }

  private final HashFunction digestFunction;

  public FileSystem() {
    this(HashFunction.MD5);
  }

  public FileSystem(HashFunction digestFunction) {
    this.digestFunction = Preconditions.checkNotNull(digestFunction);
  }

  public HashFunction getDigestFunction() {
    return digestFunction;
  }

  private enum UnixPathFactory implements PathFactory {
    INSTANCE {
      @Override
      public Path createRootPath(FileSystem filesystem) {
        return new Path(filesystem, PathFragment.ROOT_DIR, null);
      }

      @Override
      public Path createChildPath(Path parent, String childName) {
        return new Path(parent.getFileSystem(), childName, parent);
      }

      @Override
      public Path getCachedChildPathInternal(Path path, String childName) {
        return Path.getCachedChildPathInternal(path, childName, /*cacheable=*/ true);
      }
    }
  }

  /**
   * An exception thrown when attempting to resolve an ordinary file as a symlink.
   */
  protected static final class NotASymlinkException extends IOException {
    public NotASymlinkException(LocalPath path) {
      super(path.getPathString());
    }
  }

  /** Lazy-initialized on first access, always use {@link FileSystem#getRootDirectory} */
  private Path rootPath;

  /** Returns filesystem-specific path factory. */
  protected PathFactory getPathFactory() {
    return UnixPathFactory.INSTANCE;
  }

  /**
   * Returns an absolute path instance, given an absolute path name, without double slashes, .., or
   * . segments. While this method will normalize the path representation by creating a
   * structured/parsed representation, it will not cause any IO. (e.g., it will not resolve symbolic
   * links if it's a Unix file system.
   */
  public Path getPath(String pathName) {
    return getPath(PathFragment.create(pathName));
  }

  /**
   * Returns an absolute path instance, given an absolute path name, without double slashes, .., or
   * . segments. While this method will normalize the path representation by creating a
   * structured/parsed representation, it will not cause any IO. (e.g., it will not resolve symbolic
   * links if it's a Unix file system.
   */
  public final Path getPath(PathFragment pathName) {
    if (!pathName.isAbsolute()) {
      throw new IllegalArgumentException(pathName.getPathString()  + " (not an absolute path)");
    }
    return getRootDirectory().getRelative(pathName);
  }

  /**
   * Returns a path representing the root directory of the current file system.
   */
  public final Path getRootDirectory() {
    if (rootPath == null) {
      synchronized (this) {
        if (rootPath == null) {
          rootPath = getPathFactory().createRootPath(this);
        }
      }
    }
    return rootPath;
  }

  /**
   * Returns whether or not the FileSystem supports modifications of files and file entries.
   *
   * <p>Returns true if FileSystem supports the following:
   *
   * <ul>
   *   <li>{@link #setWritable(LocalPath, boolean)}
   *   <li>{@link #setExecutable(LocalPath, boolean)}
   * </ul>
   *
   * The above calls will result in an {@link UnsupportedOperationException} on a FileSystem where
   * this method returns {@code false}.
   */
  public abstract boolean supportsModifications(LocalPath path);

  /**
   * Returns whether or not the FileSystem supports symbolic links.
   *
   * <p>Returns true if FileSystem supports the following:
   *
   * <ul>
   *   <li>{@link #createSymbolicLink(LocalPath, String)}
   *   <li>{@link #getFileSize(LocalPath, boolean)} where {@code followSymlinks=false}
   *   <li>{@link #getLastModifiedTime(LocalPath, boolean)} where {@code followSymlinks=false}
   *   <li>{@link #readSymbolicLink(LocalPath)} where the link points to a non-existent file
   * </ul>
   *
   * The above calls may result in an {@link UnsupportedOperationException} on a FileSystem where
   * this method returns {@code false}. The implementation can try to emulate these calls at its own
   * discretion.
   */
  public abstract boolean supportsSymbolicLinksNatively(LocalPath path);

  /**
   * Returns whether or not the FileSystem supports hard links.
   *
   * <p>Returns true if FileSystem supports the following:
   *
   * <ul>
   *   <li>{@link #createFSDependentHardLink(LocalPath, LocalPath)}
   * </ul>
   *
   * The above calls may result in an {@link UnsupportedOperationException} on a FileSystem where
   * this method returns {@code false}. The implementation can try to emulate these calls at its own
   * discretion.
   */
  protected abstract boolean supportsHardLinksNatively(LocalPath path);

  /***
   * Returns true if file path is case-sensitive on this file system. Default is true.
   */
  public abstract boolean isFilePathCaseSensitive();

  /**
   * Returns the type of the file system path belongs to.
   *
   * <p>The string returned is obtained directly from the operating system, so it's a best guess in
   * absence of a guaranteed api.
   *
   * <p>This implementation uses <code>/proc/mounts</code> to determine the file system type.
   */
  public String getFileSystemType(LocalPath path) {
    String fileSystem = "unknown";
    int bestMountPointSegmentCount = -1;
    try {
      LocalPath canonicalPath = resolveSymbolicLinks(path);
      LocalPath mountTable = path.getRelative("/proc/mounts");
      try (InputStreamReader reader =
          new InputStreamReader(getInputStream(mountTable), ISO_8859_1)) {
        for (String line : CharStreams.readLines(reader)) {
          String[] words = line.split("\\s+");
          if (words.length >= 3) {
            if (!words[1].startsWith("/")) {
              continue;
            }
            LocalPath mountPoint = LocalPath.create(words[1]);
            int segmentCount = mountPoint.split().size();
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
  public abstract boolean createDirectory(LocalPath path) throws IOException;

  /**
   * Returns the size in bytes of the file denoted by {@code path}. See {@link
   * Path#getFileSize(Symlinks)} for specification.
   *
   * <p>Note: for <@link FileSystem>s where {@link #supportsSymbolicLinksNatively(LocalPath)}
   * returns false, this method will throw an {@link UnsupportedOperationException} if {@code
   * followSymLinks=false}.
   */
  protected abstract long getFileSize(LocalPath path, boolean followSymlinks) throws IOException;

  /** Deletes the file denoted by {@code path}. See {@link Path#delete} for specification. */
  public abstract boolean delete(LocalPath path) throws IOException;

  /**
   * Returns the last modification time of the file denoted by {@code path}. See {@link
   * Path#getLastModifiedTime(Symlinks)} for specification.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsSymbolicLinksNatively(LocalPath)}
   * returns false, this method will throw an {@link UnsupportedOperationException} if {@code
   * followSymLinks=false}.
   */
  protected abstract long getLastModifiedTime(LocalPath path, boolean followSymlinks)
      throws IOException;

  /**
   * Sets the last modification time of the file denoted by {@code path}. See {@link
   * Path#setLastModifiedTime} for specification.
   */
  public abstract void setLastModifiedTime(LocalPath path, long newTime) throws IOException;

  /**
   * Returns value of the given extended attribute name or null if attribute does not exist or file
   * system does not support extended attributes. Follows symlinks.
   *
   * <p>Default implementation assumes that file system does not support extended attributes and
   * always returns null. Specific file system implementations should override this method if they
   * do provide support for extended attributes.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if any, or null if no such
   *     attribute is defined (ENODATA) or file system does not support extended attributes at all.
   * @throws IOException if the call failed for any other reason.
   */
  public byte[] getxattr(LocalPath path, String name) throws IOException {
    return null;
  }

  /**
   * Gets a fast digest for the given path and hash function type, or {@code null} if there isn't
   * one available or the filesystem doesn't support them. This digest should be suitable for
   * detecting changes to the file.
   */
  protected byte[] getFastDigest(LocalPath path, HashFunction hashFunction) throws IOException {
    return null;
  }

  /**
   * Gets a fast digest for the given path, or {@code null} if there isn't one available or the
   * filesystem doesn't support them. This digest should be suitable for detecting changes to the
   * file.
   */
  protected final byte[] getFastDigest(LocalPath path) throws IOException {
    return getFastDigest(path, digestFunction);
  }

  /**
   * Returns whether the given digest is a valid digest for the default digest function.
   */
  public boolean isValidDigest(byte[] digest) {
    return digestFunction.isValidDigest(digest);
  }

  /**
   * Returns the digest of the file denoted by the path, following symbolic links, for the given
   * hash digest function.
   *
   * @return a new byte array containing the file's digest
   * @throws IOException if the digest could not be computed for any reason
   *     <p>Subclasses may (and do) optimize this computation for particular digest functions.
   */
  protected byte[] getDigest(LocalPath path, HashFunction hashFunction) throws IOException {
    return new ByteSource() {
      @Override
      public InputStream openStream() throws IOException {
        return getInputStream(path);
      }
    }.hash(hashFunction.getHash()).asBytes();
  }

  /**
   * Returns the digest of the file denoted by the path, following symbolic links.
   *
   * @return a new byte array containing the file's digest
   * @throws IOException if the digest could not be computed for any reason
   */
  protected final byte[] getDigest(LocalPath path) throws IOException {
    return getDigest(path, digestFunction);
  }

  /**
   * Returns true if "path" denotes an existing symbolic link. See {@link Path#isSymbolicLink} for
   * specification.
   */
  protected abstract boolean isSymbolicLink(LocalPath path);

  /**
   * Appends a single regular path segment 'child' to 'dir', recursively resolving symbolic links in
   * 'child'. 'dir' must be canonical. 'maxLinks' is the maximum number of symbolic links that may
   * be traversed before it gives up (the Linux kernel uses 32).
   *
   * <p>(This method does not need to be synchronized; but the result may be stale in the case of
   * concurrent modification.)
   *
   * @throws IOException if 'dir' is not an existing directory; or if stat(child) fails for any
   *     reason, or if 'child' is a symlink and readlink(child) fails for any reason (e.g. ENOENT,
   *     EACCES), or if the chain of symbolic links exceeds 'maxLinks'.
   */
  protected final LocalPath appendSegment(LocalPath dir, String child, int maxLinks)
      throws IOException {
    LocalPath naive = dir.getRelative(child);

    String linkTarget = resolveOneLink(naive);
    if (linkTarget == null) {
      return naive; // regular file or directory
    }

    if (maxLinks-- == 0) {
      throw new IOException(naive + " (Too many levels of symbolic links)");
    }
    LocalPath linkTargetPath = LocalPath.create(linkTarget);
    if (linkTargetPath.isAbsolute()) {
      dir = linkTargetPath.getDrive();
    }
    for (String name : linkTargetPath.split()) {
      if (name.equals(".") || name.isEmpty()) {
        // no-op
      } else if (name.equals("..")) {
        LocalPath parent = dir.getParentDirectory();
        // root's parent is root, when canonicalizing, so this is a no-op.
        if (parent != null) { dir = parent; }
      } else {
        dir = appendSegment(dir, name, maxLinks);
      }
    }
    return dir;
  }

  /**
   * Helper method of {@link #resolveSymbolicLinks(LocalPath)}. This method encapsulates the I/O
   * component of a full canonicalization operation. Subclasses can (and do) provide more efficient
   * implementations.
   *
   * <p>(This method does not need to be synchronized; but the result may be stale in the case of
   * concurrent modification.)
   *
   * @param path a path, of which all but the last segment is guaranteed to be canonical
   * @return {@link #readSymbolicLink} iff path is a symlink or null iff path exists but is not a
   *     symlink
   * @throws IOException if the file did not exist, or a parent directory could not be searched
   */
  protected String resolveOneLink(LocalPath path) throws IOException {
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
   * Returns the canonical path for the given path. See {@link Path#resolveSymbolicLinks} for
   * specification.
   */
  protected LocalPath resolveSymbolicLinks(LocalPath path) throws IOException {
    LocalPath parentNode = path.getParentDirectory();
    return parentNode == null
        ? path // (root)
        : appendSegment(resolveSymbolicLinks(parentNode), path.getBaseName(), 32);
  }

  /**
   * Returns the status of a file. See {@link Path#stat(Symlinks)} for specification.
   *
   * <p>The default implementation of this method is a "lazy" one, based on other accessor methods
   * such as {@link #isFile}, etc. Subclasses may provide more efficient specializations. However,
   * we still try to follow Unix-like semantics of failing fast in case of non-existent files (or in
   * case of permission issues).
   */
  protected FileStatus stat(LocalPath path, boolean followSymlinks) throws IOException {
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

  /** Like stat(), but returns null on failures instead of throwing. */
  protected FileStatus statNullable(LocalPath path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks);
    } catch (IOException e) {
      return null;
    }
  }

  /**
   * Like {@link #stat}, but returns null if the file is not found (corresponding to {@code ENOENT}
   * or {@code ENOTDIR} in Unix's stat(2) function) instead of throwing. Note that this
   * implementation does <i>not</i> successfully catch {@code ENOTDIR} exceptions. If the
   * instantiated filesystem can catch such errors, it should override this method to do so.
   */
  protected FileStatus statIfFound(LocalPath path, boolean followSymlinks) throws IOException {
    try {
      return stat(path, followSymlinks);
    } catch (FileNotFoundException e) {
      return null;
    }
  }

  /**
   * Returns true iff {@code path} denotes an existing directory. See {@link
   * Path#isDirectory(Symlinks)} for specification.
   */
  protected abstract boolean isDirectory(LocalPath path, boolean followSymlinks);

  /**
   * Returns true iff {@code path} denotes an existing regular or special file. See {@link
   * Path#isFile(Symlinks)} for specification.
   */
  protected abstract boolean isFile(LocalPath path, boolean followSymlinks);

  /**
   * Returns true iff {@code path} denotes a special file. See {@link Path#isSpecialFile(Symlinks)}
   * for specification.
   */
  protected abstract boolean isSpecialFile(LocalPath path, boolean followSymlinks);

  /**
   * Creates a symbolic link. See {@link Path#createSymbolicLink(Path)} for specification.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsSymbolicLinksNatively(LocalPath)}
   * returns false, this method will throw an {@link UnsupportedOperationException}
   */
  protected abstract void createSymbolicLink(LocalPath linkPath, String targetFragment)
      throws IOException;

  /**
   * Returns the target of a symbolic link. See {@link Path#readSymbolicLink} for specification.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsSymbolicLinksNatively(LocalPath)}
   * returns false, this method will throw an {@link UnsupportedOperationException} if the link
   * points to a non-existent file.
   *
   * @throws NotASymlinkException if the current path is not a symbolic link
   * @throws IOException if the contents of the link could not be read for any reason.
   */
  protected abstract String readSymbolicLink(LocalPath path) throws IOException;

  /**
   * Returns the target of a symbolic link, under the assumption that the given path is indeed a
   * symbolic link (this assumption permits efficient implementations). See {@link
   * Path#readSymbolicLinkUnchecked} for specification.
   *
   * @throws IOException if the contents of the link could not be read for any reason.
   */
  protected String readSymbolicLinkUnchecked(LocalPath path) throws IOException {
    return readSymbolicLink(path);
  }

  /** Returns true iff this path denotes an existing file of any kind. Follows symbolic links. */
  public boolean exists(LocalPath path) {
    return exists(path, true);
  }

  /**
   * Returns true iff {@code path} denotes an existing file of any kind. See {@link
   * Path#exists(Symlinks)} for specification.
   */
  protected abstract boolean exists(LocalPath path, boolean followSymlinks);

  /**
   * Returns a collection containing the names of all entities within the directory denoted by the
   * {@code path}.
   *
   * @throws IOException if there was an error reading the directory entries
   */
  protected abstract Collection<String> getDirectoryEntries(LocalPath path) throws IOException;

  protected static Dirent.Type direntFromStat(FileStatus stat) {
    if (stat == null) {
      return Type.UNKNOWN;
    } else if (stat.isSpecialFile()) {
        return Type.UNKNOWN;
    } else if (stat.isFile()) {
      return Type.FILE;
    } else if (stat.isDirectory()) {
      return Type.DIRECTORY;
    } else if (stat.isSymbolicLink()) {
      return Type.SYMLINK;
    } else {
      return Type.UNKNOWN;
    }
  }

  /**
   * Returns a Dirents structure, listing the names of all entries within the directory {@code
   * path}, plus their types (file, directory, other).
   *
   * @param followSymlinks whether to follow symlinks when determining the file types of individual
   *     directory entries. No matter the value of this parameter, symlinks are followed when
   *     resolving the directory whose entries are to be read.
   * @throws IOException if there was an error reading the directory entries
   */
  protected Collection<Dirent> readdir(LocalPath path, boolean followSymlinks) throws IOException {
    Collection<String> children = getDirectoryEntries(path);
    List<Dirent> dirents = Lists.newArrayListWithCapacity(children.size());
    for (String child : children) {
      LocalPath childPath = path.getRelative(child);
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
  protected abstract boolean isReadable(LocalPath path) throws IOException;

  /**
   * Sets the file to readable (if the argument is true) or non-readable (if the argument is false)
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(LocalPath)} returns false
   * or which do not support unreadable files, this method will throw an {@link
   * UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  protected abstract void setReadable(LocalPath path, boolean readable) throws IOException;

  /**
   * Returns true iff the file represented by {@code path} is writable.
   *
   * @throws IOException if there was an error reading the file's metadata
   */
  protected abstract boolean isWritable(LocalPath path) throws IOException;

  /**
   * Sets the file to writable (if the argument is true) or non-writable (if the argument is false)
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(LocalPath)} returns false,
   * this method will throw an {@link UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  public abstract void setWritable(LocalPath path, boolean writable) throws IOException;

  /**
   * Returns true iff the file represented by the path is executable.
   *
   * @throws IOException if there was an error reading the file's metadata
   */
  protected abstract boolean isExecutable(LocalPath path) throws IOException;

  /**
   * Sets the file to executable, if the argument is true. It is currently not supported to unset
   * the executable status of a file, so {code executable=false} yields an {@link
   * UnsupportedOperationException}.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(LocalPath)} returns false,
   * this method will throw an {@link UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  protected abstract void setExecutable(LocalPath path, boolean executable) throws IOException;

  /**
   * Sets the file permissions. If permission changes on this {@link FileSystem} are slow (e.g. one
   * syscall per change), this method should aim to be faster than setting each permission
   * individually. If this {@link FileSystem} does not support group or others permissions, those
   * bits will be ignored.
   *
   * <p>Note: for {@link FileSystem}s where {@link #supportsModifications(LocalPath)} returns false,
   * this method will throw an {@link UnsupportedOperationException}.
   *
   * @throws IOException if there was an error reading or writing the file's metadata
   */
  protected void chmod(LocalPath path, int mode) throws IOException {
    setReadable(path, (mode & 0400) != 0);
    setWritable(path, (mode & 0200) != 0);
    setExecutable(path, (mode & 0100) != 0);
  }

  /**
   * Creates an InputStream accessing the file denoted by the path.
   *
   * @throws IOException if there was an error opening the file for reading
   */
  protected abstract InputStream getInputStream(LocalPath path) throws IOException;

  /**
   * Creates an OutputStream accessing the file denoted by path.
   *
   * @throws IOException if there was an error opening the file for writing
   */
  protected final OutputStream getOutputStream(LocalPath path) throws IOException {
    return getOutputStream(path, false);
  }

  /**
   * Creates an OutputStream accessing the file denoted by path.
   *
   * @param append whether to open the output stream in append mode
   * @throws IOException if there was an error opening the file for writing
   */
  protected abstract OutputStream getOutputStream(LocalPath path, boolean append)
      throws IOException;

  /**
   * Renames the file denoted by "sourceNode" to the location "targetNode". See {@link
   * Path#renameTo} for specification.
   */
  public abstract void renameTo(LocalPath sourcePath, LocalPath targetPath) throws IOException;

  /**
   * Create a new hard link file at "linkPath" for file at "originalPath".
   *
   * @param linkPath The path of the new link file to be created
   * @param originalPath The path of the original file
   * @throws IOException if the original file does not exist or the link file already exists
   */
  protected void createHardLink(LocalPath linkPath, LocalPath originalPath) throws IOException {

    if (!exists(originalPath, true)) {
      throw new FileNotFoundException(
          "File \""
              + originalPath.getBaseName()
              + "\" linked from \""
              + linkPath.getBaseName()
              + "\" does not exist");
    }

    if (exists(linkPath, true)) {
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
  protected abstract void createFSDependentHardLink(LocalPath linkPath, LocalPath originalPath)
      throws IOException;

  /**
   * Prefetch all directories and symlinks within the package rooted at "path". Enter at most
   * "maxDirs" total directories. Specializations for high-latency remote filesystems may wish to
   * implement this in order to warm the filesystem's internal caches.
   */
  protected void prefetchPackageAsync(LocalPath path, int maxDirs) {}
}
