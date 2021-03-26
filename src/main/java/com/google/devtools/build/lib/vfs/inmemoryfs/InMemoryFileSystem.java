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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.devtools.build.lib.collect.CollectionUtils.isNullOrEmpty;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.AbstractFileSystemWithCustomStat;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileAccessException;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CheckReturnValue;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;

/**
 * This class provides a complete in-memory file system.
 *
 * <p>Naming convention: we use "path" for all {@link Path} variables, since these represent *names*
 * and we use "node" or "inode" for InMemoryContentInfo variables, since these correspond to inodes
 * in the UNIX file system.
 *
 * <p>The code is structured to be as similar to the implementation of UNIX "namei" as is reasonably
 * possibly. This provides a firm reference point for many concepts and makes compatibility easier
 * to achieve.
 */
@ThreadSafe
public class InMemoryFileSystem extends AbstractFileSystemWithCustomStat {

  protected final Clock clock;

  // The root inode (a directory).
  private final InMemoryDirectoryInfo rootInode;

  // Maximum number of traversals before ELOOP is thrown.
  private static final int MAX_TRAVERSALS = 256;

  /**
   * Creates a new {@code InMemoryFileSystem} with default clock and given hash function.
   *
   * @param hashFunction the function to use for calculating digests.
   */
  public InMemoryFileSystem(DigestHashFunction hashFunction) {
    this(new JavaClock(), hashFunction);
  }

  /** Creates a new {@code InMemoryFileSystem} with the given clock and hash function. */
  public InMemoryFileSystem(Clock clock, DigestHashFunction hashFunction) {
    super(hashFunction);
    this.clock = clock;
    this.rootInode = newRootInode(clock);
  }

  private static InMemoryDirectoryInfo newRootInode(Clock clock) {
    InMemoryDirectoryInfo rootInode = new InMemoryDirectoryInfo(clock);
    rootInode.addChild(".", rootInode);
    rootInode.addChild("..", rootInode);
    return rootInode;
  }

  /** The errors that {@link InMemoryFileSystem} might issue for different sorts of IO failures. */
  protected enum Errno implements InodeOrErrno {
    ENOENT("No such file or directory"),
    EACCES("Permission denied"),
    ENOTDIR("Not a directory"),
    EEXIST("File exists"),
    EBUSY("Device or resource busy"),
    ENOTEMPTY("Directory not empty"),
    EISDIR("Is a directory"),
    ELOOP("Too many levels of symbolic links");

    private final String message;

    Errno(String message) {
      this.message = message;
    }

    @Nullable
    @Override
    public InMemoryContentInfo inode() {
      return null;
    }

    @Override
    public Errno error() {
      return this;
    }

    @Override
    public boolean isError() {
      return true;
    }

    @Override
    public InMemoryContentInfo inodeOrThrow(PathFragment path) throws IOException {
      throw exception(path);
    }

    @Override
    public String toString() {
      return message;
    }

    /**
     * Throws a new {@link IOException} for this error. The exception message contains {@code path},
     * and is consistent with the messages returned by {@link
     * com.google.devtools.build.lib.vfs.FileSystemUtils}.
     */
    public IOException exception(PathFragment path) throws IOException {
      String m = path + " (" + message + ")";
      switch (this) {
        case EACCES:
          throw new FileAccessException(m);
        case ENOENT:
          throw new FileNotFoundException(m);
        default:
          throw new IOException(m);
      }
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>If <code>/proc/mounts</code> does not exist return {@code "inmemoryfs"}.
   */
  @Override
  public String getFileSystemType(PathFragment path) {
    return exists(path.getRelative("/proc/mounts")) ? super.getFileSystemType(path) : "inmemoryfs";
  }

  /*
   ***************************************************************************
   * "Kernel" primitives: basic directory lookup primitives, in topological order.
   */

  /**
   * Unlinks the entry 'child' from its existing parent directory 'dir'. Dual to insert. This
   * succeeds even if 'child' names a non-empty directory; we need that for renameTo. 'child' must
   * be a member of its parent directory, however. Fails if the directory was read-only.
   */
  private static void unlink(InMemoryDirectoryInfo dir, String child, PathFragment errorPath)
      throws IOException {
    if (!dir.isWritable()) {
      throw Errno.EACCES.exception(errorPath);
    }
    dir.removeChild(child);
  }

  /**
   * Inserts inode 'childInode' into the existing directory 'dir' under the specified 'name'. Dual
   * to unlink. Fails if the directory was read-only.
   */
  @CheckReturnValue
  private static Errno insert(
      InMemoryDirectoryInfo dir, String child, InMemoryContentInfo childInode) {
    if (!dir.isWritable()) {
      return Errno.EACCES;
    }
    dir.addChild(child, childInode);
    return null;
  }

  private static void insert(
      InMemoryDirectoryInfo dir,
      String child,
      InMemoryContentInfo childInode,
      PathFragment errorPath)
      throws IOException {
    Errno error = insert(dir, child, childInode);
    if (error != null) {
      throw error.exception(errorPath);
    }
  }

  /**
   * Given an existing directory 'dir', looks up 'name' within it and returns its inode. May fail
   * with ENOTDIR, EACCES, ENOENT. Error messages will be reported against file 'path'.
   */
  private static InodeOrErrno directoryLookupErrno(InMemoryContentInfo dir, String name) {
    if (!dir.isDirectory()) {
      return Errno.ENOTDIR;
    }
    if (!dir.isExecutable()) {
      return Errno.EACCES;
    }
    return firstNonNull(dir.asDirectory().getChild(name), Errno.ENOENT);
  }

  protected FileInfo newFile(Clock clock, PathFragment path) {
    return new InMemoryFileInfo(clock);
  }

  /** How to handle {@link Errno#ENOENT} during {@link #pathWalkErrno}. */
  private enum OnEnoent {
    /** Halt the walk with {@link Errno#ENOENT}. */
    HALT,
    /**
     * Create a file node if at the last segment of the walk, otherwise halt with {@link
     * Errno#ENOENT}.
     */
    CREATE_FILE,
    /** Create a directory node. */
    CREATE_DIRECTORY_AND_PARENTS
  }

  /**
   * Low-level path-to-inode lookup routine. Analogous to path_walk() in many UNIX kernels. Given
   * 'path', walks the directory tree from the root, resolving all symbolic links, and returns the
   * designated inode.
   *
   * <p>ENOENT along the walk is handled according to the given {@link OnEnoent}.
   *
   * <p>May fail with ENOTDIR, ENOENT, EACCES, ELOOP.
   */
  private synchronized InodeOrErrno pathWalkErrno(PathFragment path, OnEnoent behavior) {
    Iterator<String> it = path.segments().iterator();

    // Prepend the Windows drive if there is one.
    if (path.getDriveStrLength() > 1) {
      it = Iterators.concat(Iterators.singletonIterator(path.getDriveStr()), it);
    }

    InMemoryContentInfo inode = rootInode;
    int traversals = 0;

    // Stack of symlink targets. Lazily initialized because we probably won't see any.
    Deque<String> symlinks = null;

    while (it.hasNext() || !isNullOrEmpty(symlinks)) {
      traversals++;

      String name = !isNullOrEmpty(symlinks) ? symlinks.pop() : it.next();

      InodeOrErrno childOrError = directoryLookupErrno(inode, name);

      InMemoryContentInfo child;
      if (!childOrError.isError()) {
        child = childOrError.inode();
      } else if (childOrError.error() == Errno.ENOENT && behavior != OnEnoent.HALT) {
        InMemoryDirectoryInfo parent = inode.asDirectory();
        Errno error;
        if (behavior == OnEnoent.CREATE_DIRECTORY_AND_PARENTS) {
          // ENOENT anywhere with Create.DIRECTORY_AND_PARENTS => create a new directory.
          InMemoryDirectoryInfo newDir = new InMemoryDirectoryInfo(clock);
          error = insertChildDirectory(parent, newDir, name);
          child = newDir;
        } else if (!it.hasNext() && isNullOrEmpty(symlinks)) {
          // ENOENT on last segment with Create.FILE => create a new file.
          child = newFile(clock, path);
          error = insert(parent, name, child);
        } else {
          return childOrError;
        }
        if (error != null) {
          return error;
        }
      } else {
        return childOrError;
      }

      if (!child.isSymbolicLink()) {
        inode = child;
      } else {
        PathFragment linkTarget = ((InMemoryLinkInfo) child).getNormalizedLinkContent();
        if (linkTarget.isAbsolute()) {
          inode = rootInode;
        }
        if (traversals > MAX_TRAVERSALS) {
          return Errno.ELOOP;
        }

        List<String> segments = linkTarget.splitToListOfSegments(); // May include ".." segments.
        if (symlinks == null) {
          symlinks = new ArrayDeque<>(segments);
        } else {
          for (int ii = segments.size() - 1; ii >= 0; --ii) {
            symlinks.push(segments.get(ii));
          }
        }
        // Push Windows drive if there is one.
        if (linkTarget.getDriveStrLength() > 1) {
          symlinks.push(linkTarget.getDriveStr());
        }
      }
    }
    return inode;
  }

  /**
   * Given 'path', returns the existing directory inode it designates, following symbolic links.
   *
   * <p>May fail with ENOTDIR, or any exception from pathWalk.
   */
  private InodeOrErrno getDirectoryErrno(PathFragment path) {
    InodeOrErrno dirInfoOrError = pathWalkErrno(path, OnEnoent.HALT);
    if (dirInfoOrError.isError()) {
      return dirInfoOrError;
    }
    return dirInfoOrError.inode().isDirectory() ? dirInfoOrError : Errno.ENOTDIR;
  }

  /**
   * Given 'path', returns the existing directory inode it designates, following symbolic links.
   *
   * <p>May fail with ENOTDIR, or any exception from pathWalk.
   */
  private InMemoryDirectoryInfo getDirectory(PathFragment path) throws IOException {
    return getDirectoryErrno(path).inodeOrThrow(path).asDirectory();
  }

  /** Helper method for stat and inodeStat: return the path's (no symlink-followed) stat. */
  private synchronized InodeOrErrno noFollowStatErrno(PathFragment path) {
    InodeOrErrno dirInfoOrError = getDirectoryErrno(path.getParentDirectory());
    if (dirInfoOrError.isError()) {
      return dirInfoOrError;
    }
    return directoryLookupErrno(dirInfoOrError.inode(), baseNameOrWindowsDrive(path));
  }

  /**
   * Given 'path', returns the existing inode it designates, optionally following symbolic links.
   * Analogous to UNIX stat(2)/lstat(2), except that it returns a mutable inode we can modify
   * directly.
   */
  @Override
  public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return inodeStatErrno(path, followSymlinks).inodeOrThrow(path);
  }

  @Override
  @Nullable
  public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    InodeOrErrno inodeOrErrno = inodeStatErrno(path, followSymlinks);
    if (!inodeOrErrno.isError()) {
      return inodeOrErrno.inode();
    }
    Errno errorCode = inodeOrErrno.error();
    if (errorCode == Errno.ENOENT || errorCode == Errno.ENOTDIR) {
      return null;
    }
    throw errorCode.exception(path);
  }

  @Override
  @Nullable
  protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    return inodeStatErrno(path, followSymlinks).inode();
  }

  /** Version of stat that returns an InodeOrErrno of the input path. */
  @CheckReturnValue
  protected InodeOrErrno inodeStatErrno(PathFragment path, boolean followSymlinks) {
    if (followSymlinks) {
      return pathWalkErrno(path, OnEnoent.HALT);
    }
    return isRootDirectory(path) ? rootInode : noFollowStatErrno(path);
  }

  private InMemoryContentInfo inodeStat(PathFragment path, boolean followSymlinks)
      throws IOException {
    return inodeStatErrno(path, followSymlinks).inodeOrThrow(path);
  }

  /*
   ***************************************************************************
   * FileSystem methods
   */

  /**
   * This is a helper routing for {@link #resolveSymbolicLinks(PathFragment)}, i.e. the "user-mode"
   * routing for canonicalizing paths. It is analogous to the code in glibc's realpath(3).
   *
   * <p>Just like realpath, resolveSymbolicLinks requires a quadratic number of directory lookups: n
   * path segments are statted, and each stat requires a linear amount of work in the "kernel"
   * routine.
   */
  @Override
  protected PathFragment resolveOneLink(PathFragment path) throws IOException {
    // Beware, this seemingly simple code belies the complex specification of
    // FileSystem.resolveOneLink().
    InMemoryContentInfo status = inodeStat(path, false);
    return status.isSymbolicLink() ? ((InMemoryLinkInfo) status).getLinkContent() : null;
  }

  @Override
  protected boolean exists(PathFragment path, boolean followSymlinks) {
    return statNullable(path, followSymlinks) != null;
  }

  @Override
  protected boolean isReadable(PathFragment path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    return status.isReadable();
  }

  @Override
  protected synchronized void setReadable(PathFragment path, boolean readable) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    status.setReadable(readable);
  }

  @Override
  protected boolean isWritable(PathFragment path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    return status.isWritable();
  }

  @Override
  public synchronized void setWritable(PathFragment path, boolean writable) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    status.setWritable(writable);
  }

  @Override
  protected boolean isExecutable(PathFragment path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    return status.isExecutable();
  }

  @Override
  protected synchronized void setExecutable(PathFragment path, boolean executable)
      throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    status.setExecutable(executable);
  }

  @Override
  public boolean supportsModifications(PathFragment path) {
    return true;
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return true;
  }

  @Override
  public boolean supportsHardLinksNatively(PathFragment path) {
    return true;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return OS.getCurrent() != OS.WINDOWS;
  }

  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    if (isRootDirectory(path)) {
      throw Errno.EACCES.exception(path);
    }

    PathFragment parentDir = path.getParentDirectory();
    String name = baseNameOrWindowsDrive(path);
    Errno error;
    synchronized (this) {
      InMemoryDirectoryInfo parent = getDirectory(parentDir);
      InMemoryContentInfo child = parent.getChild(name);
      if (child != null) { // already exists
        if (!child.isDirectory()) {
          throw Errno.EEXIST.exception(path);
        }
        return false;
      }
      error = insertChildDirectory(parent, new InMemoryDirectoryInfo(clock), name);
    }
    if (error != null) {
      throw error.exception(path);
    }
    return true;
  }

  @Nullable
  private static Errno insertChildDirectory(
      InMemoryDirectoryInfo parent, InMemoryDirectoryInfo newDir, String name) {
    newDir.addChild(".", newDir);
    newDir.addChild("..", parent);
    return insert(parent, name, newDir);
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    InMemoryContentInfo result =
        pathWalkErrno(path, OnEnoent.CREATE_DIRECTORY_AND_PARENTS).inodeOrThrow(path);
    if (!result.isDirectory()) {
      throw new IOException("Not a directory: " + path);
    }
  }

  @Override
  protected void createSymbolicLink(PathFragment path, PathFragment targetFragment)
      throws IOException {
    if (isRootDirectory(path)) {
      throw Errno.EACCES.exception(path);
    }

    synchronized (this) {
      InMemoryDirectoryInfo parent = getDirectory(path.getParentDirectory());
      if (parent.getChild(baseNameOrWindowsDrive(path)) != null) {
        throw Errno.EEXIST.exception(path);
      }
      insert(
          parent, baseNameOrWindowsDrive(path), new InMemoryLinkInfo(clock, targetFragment), path);
    }
  }

  @Override
  protected PathFragment readSymbolicLink(PathFragment path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, false);
    if (status.isSymbolicLink()) {
      Preconditions.checkState(status instanceof InMemoryLinkInfo, status);
      return ((InMemoryLinkInfo) status).getLinkContent();
    }
    throw new NotASymlinkException(path);
  }

  @Override
  protected long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getSize();
  }

  @Override
  protected synchronized Collection<String> getDirectoryEntries(PathFragment path)
      throws IOException {
    InMemoryDirectoryInfo dirInfo = getDirectory(path);
    if (!dirInfo.isReadable()) {
      throw Errno.EACCES.exception(path);
    }

    Collection<String> allChildren = dirInfo.getAllChildren();
    List<String> result = new ArrayList<>(allChildren.size());
    for (String child : allChildren) {
      if (!child.equals(".") && !child.equals("..")) {
        result.add(child);
      }
    }
    return result;
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    if (isRootDirectory(path)) {
      throw Errno.EBUSY.exception(path);
    }

    synchronized (this) {
      if (!exists(path, /*followSymlinks=*/ false)) {
        return false;
      }
      InMemoryDirectoryInfo parent = getDirectory(path.getParentDirectory());
      InMemoryContentInfo child = parent.getChild(baseNameOrWindowsDrive(path));
      if (child.isDirectory() && child.getSize() > 2) {
        throw Errno.ENOTEMPTY.exception(path);
      }
      unlink(parent, baseNameOrWindowsDrive(path), path);
      return true;
    }
  }

  @Override
  protected long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  public synchronized void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    status.setLastModifiedTime(newTime == -1L ? clock.currentTimeMillis() : newTime);
  }

  @Override
  protected synchronized InputStream getInputStream(PathFragment path) throws IOException {
    return statFile(path).getInputStream();
  }

  @Override
  protected synchronized ReadableByteChannel createReadableByteChannel(PathFragment path)
      throws IOException {
    return statFile(path).createReadableByteChannel();
  }

  @Override
  protected synchronized byte[] getFastDigest(PathFragment path) throws IOException {
    return statFile(path).getFastDigest();
  }

  private FileInfo statFile(PathFragment path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, /*followSymlinks=*/ true);
    if (status.isDirectory()) {
      throw Errno.EISDIR.exception(path);
    }
    if (!status.isReadable()) {
      throw Errno.EACCES.exception(path);
    }
    Preconditions.checkState(status instanceof FileInfo, status);
    return (FileInfo) status;
  }

  @Override
  @Nullable
  public synchronized byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    InMemoryContentInfo status = inodeStat(path, followSymlinks);
    if (status.isDirectory()) {
      throw Errno.EISDIR.exception(path);
    }
    if (!isReadable(path)) {
      throw Errno.EACCES.exception(path);
    }
    if (!followSymlinks && status.isSymbolicLink()) {
      return null; // xattr on symlinks not supported.
    }
    Preconditions.checkState(status instanceof FileInfo, status);
    return ((FileInfo) status).getxattr(name);
  }

  /** Creates a new file at the given path and returns its inode. */
  protected InMemoryContentInfo getOrCreateWritableInode(PathFragment path) throws IOException {
    // open(WR_ONLY) of a dangling link writes through the link.  That means
    // that the usual path lookup operations have to behave differently when
    // resolving a path with the intent to create it: instead of failing with
    // ENOENT they have to return an open file.  This is exactly how UNIX
    // kernels do it, which is what we're trying to emulate.
    InMemoryContentInfo child = pathWalkErrno(path, OnEnoent.CREATE_FILE).inodeOrThrow(path);
    if (child.isDirectory()) {
      throw Errno.EISDIR.exception(path);
    }
    if (!child.isWritable()) {
      throw Errno.EACCES.exception(path);
    }
    return child;
  }

  @Override
  protected synchronized OutputStream getOutputStream(PathFragment path, boolean append)
      throws IOException {
    InMemoryContentInfo status = getOrCreateWritableInode(path);
    return ((FileInfo) status).getOutputStream(append);
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    if (isRootDirectory(sourcePath)) {
      throw Errno.EACCES.exception(sourcePath);
    }
    if (isRootDirectory(targetPath)) {
      throw Errno.EACCES.exception(targetPath);
    }
    synchronized (this) {
      InMemoryDirectoryInfo sourceParent = getDirectory(sourcePath.getParentDirectory());
      InMemoryDirectoryInfo targetParent = getDirectory(targetPath.getParentDirectory());

      InMemoryContentInfo sourceInode = sourceParent.getChild(baseNameOrWindowsDrive(sourcePath));
      if (sourceInode == null) {
        throw Errno.ENOENT.exception(sourcePath);
      }
      InMemoryContentInfo targetInode = targetParent.getChild(baseNameOrWindowsDrive(targetPath));

      unlink(sourceParent, baseNameOrWindowsDrive(sourcePath), sourcePath);
      try {
        // TODO(bazel-team): (2009) test with symbolic links.

        // Precondition checks:
        if (targetInode != null) { // already exists
          if (targetInode.isDirectory()) {
            if (!sourceInode.isDirectory()) {
              throw new IOException(sourcePath + " -> " + targetPath + " (" + Errno.EISDIR + ")");
            }
            if (targetInode.getSize() > 2) {
              throw Errno.ENOTEMPTY.exception(targetPath);
            }
          } else if (sourceInode.isDirectory()) {
            throw new IOException(sourcePath + " -> " + targetPath + " (" + Errno.ENOTDIR + ")");
          }
          unlink(targetParent, baseNameOrWindowsDrive(targetPath), targetPath);
        }
        insert(targetParent, baseNameOrWindowsDrive(targetPath), sourceInode, targetPath);
      } catch (IOException e) {
        insert(
            sourceParent,
            baseNameOrWindowsDrive(sourcePath),
            sourceInode,
            sourcePath); // restore source
        throw e;
      }
    }
  }

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {

    // Same check used when creating a symbolic link
    if (isRootDirectory(originalPath)) {
      throw Errno.EACCES.exception(originalPath);
    }

    synchronized (this) {
      InMemoryDirectoryInfo linkParent = getDirectory(linkPath.getParentDirectory());
      // Same check used when creating a symbolic link
      if (linkParent.getChild(baseNameOrWindowsDrive(linkPath)) != null) {
        throw Errno.EEXIST.exception(linkPath);
      }
      insert(
          linkParent,
          baseNameOrWindowsDrive(linkPath),
          getDirectory(originalPath.getParentDirectory())
              .getChild(baseNameOrWindowsDrive(originalPath)),
          linkPath);
    }
  }

  /**
   * On Unix the root directory is "/". On Windows there isn't one, so we reach null from
   * getParentDirectory.
   */
  private static boolean isRootDirectory(@Nullable PathFragment path) {
    return path == null || path.getPathString().equals("/");
  }

  /**
   * Returns either the base name of the path, or the drive (if referring to a Windows drive).
   *
   * <p>This allows the file system to treat windows drives much like directories.
   */
  private static String baseNameOrWindowsDrive(PathFragment path) {
    String name = path.getBaseName();
    return !name.isEmpty() ? name : path.getDriveStr();
  }

  /** Represents either an {@link Errno} or an {@link InMemoryContentInfo}. */
  protected interface InodeOrErrno {

    @Nullable
    InMemoryContentInfo inode();

    @Nullable
    Errno error();

    boolean isError();

    /**
     * Returns the underlying {@link InMemoryContentInfo} unless this {@link #isError}, in which
     * case {@link IOException} is thrown, using the given path to construct an error message.
     */
    InMemoryContentInfo inodeOrThrow(PathFragment path) throws IOException;
  }
}
