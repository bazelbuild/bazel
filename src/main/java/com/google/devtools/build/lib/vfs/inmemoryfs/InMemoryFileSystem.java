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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
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
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Stack;
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
   * Creates a new InMemoryFileSystem with default clock and given hash function.
   *
   * @param hashFunction the function to use for calculating digests.
   */
  public InMemoryFileSystem(DigestHashFunction hashFunction) {
    this(new JavaClock(), hashFunction);
  }

  /**
   * Creates a new InMemoryFileSystem with the given clock and hash function.
   */
  public InMemoryFileSystem(Clock clock, DigestHashFunction hashFunction) {
    super(hashFunction);
    this.clock = clock;
    this.rootInode = newRootInode(clock);
  }

  /**
   * Creates a new InMemoryFileSystem with default clock and hash function.
   */
  @VisibleForTesting
  public InMemoryFileSystem() {
    this(new JavaClock());
  }

  /**
   * Creates a new InMemoryFileSystem.
   */
  @VisibleForTesting
  public InMemoryFileSystem(Clock clock) {
    this(clock, DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
  }

  private static InMemoryDirectoryInfo newRootInode(Clock clock) {
    InMemoryDirectoryInfo rootInode = new InMemoryDirectoryInfo(clock);
    rootInode.addChild(".", rootInode);
    rootInode.addChild("..", rootInode);
    return rootInode;
  }

  /**
   * Given a path that's normalized (no ".." or "." segments), except for a possible prefix sequence
   * of contiguous ".." segments, returns the size of that prefix sequence.
   *
   * <p>Example allowed inputs: "/absolute/path", "relative/path", "../../relative/path". Example
   * disallowed inputs: "/absolute/path/../path2", "relative/../path", "../relative/../p".
   */
  private int leadingParentReferences(PathFragment normalizedPath) {
    int leadingParentReferences = 0;
    for (int i = 0;
        i < normalizedPath.segmentCount() && normalizedPath.getSegment(i).equals("..");
        i++) {
      leadingParentReferences++;
    }
    return leadingParentReferences;
  }

  /**
   * The errors that {@link InMemoryFileSystem} might issue for different sorts of IO failures.
   */
  public enum Error {
    ENOENT("No such file or directory"),
    EACCES("Permission denied"),
    ENOTDIR("Not a directory"),
    EEXIST("File exists"),
    EBUSY("Device or resource busy"),
    ENOTEMPTY("Directory not empty"),
    EISDIR("Is a directory"),
    ELOOP("Too many levels of symbolic links");

    private final String message;

    Error(String message) {
      this.message = message;
    }

    @Override
    public String toString() {
      return message;
    }

    public InodeOrErrno asInodeOrErrno() {
      return InodeOrErrno.createError(this);
    }

    /** Implemented by exceptions that contain the extra info of which Error caused them. */
    private interface WithError {
      Error getError();
    }

    /**
     * The exceptions below extend their parent classes in order to additionally store the error
     * that caused them. However, they must impersonate their parents to any outside callers,
     * including in their toString() method, which prints the class name followed by the exception
     * method. This method returns the same value as the toString() method of a {@link Throwable}'s
     * parent would, so that the child class can have the same toString() value.
     */
    private static String parentThrowableToString(Throwable obj) {
      String s = obj.getClass().getSuperclass().getName();
      String message = obj.getLocalizedMessage();
      return (message != null) ? (s + ": " + message) : s;
    }

    private static class IOExceptionWithError extends IOException implements WithError {
      private final Error errorCode;

      private IOExceptionWithError(String message, Error errorCode) {
        super(message);
        this.errorCode = errorCode;
      }

      @Override
      public Error getError() {
        return errorCode;
      }

      @Override
      public String toString() {
        return parentThrowableToString(this);
      }
    }


    private static class FileNotFoundExceptionWithError
        extends FileNotFoundException implements WithError {
      private final Error errorCode;

      private FileNotFoundExceptionWithError(String message, Error errorCode) {
        super(message);
        this.errorCode = errorCode;
      }

      @Override
      public Error getError() {
        return errorCode;
      }

      @Override
      public String toString() {
        return parentThrowableToString(this);
      }
    }


    private static class FileAccessExceptionWithError
        extends FileAccessException implements WithError {
      private final Error errorCode;

      private FileAccessExceptionWithError(String message, Error errorCode) {
        super(message);
        this.errorCode = errorCode;
      }

      @Override
      public Error getError() {
        return errorCode;
      }

      @Override
      public String toString() {
        return parentThrowableToString(this);
      }
    }

    /**
     * Returns a new IOException for the error. The exception message
     * contains 'path', and is consistent with the messages returned by
     * c.g.common.unix.FilesystemUtils.
     */
    public IOException exception(Path path) throws IOException {
      String m = path + " (" + message + ")";
      if (this == EACCES) {
        throw new FileAccessExceptionWithError(m, this);
      } else if (this == ENOENT) {
        throw new FileNotFoundExceptionWithError(m, this);
      } else {
        throw new IOExceptionWithError(m, this);
      }
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>If <code>/proc/mounts</code> does not exist return {@code "inmemoryfs"}.
   */
  @Override
  public String getFileSystemType(Path path) {
    return path.getRelative("/proc/mounts").exists() ? super.getFileSystemType(path) : "inmemoryfs";
  }

  /****************************************************************************
   * "Kernel" primitives: basic directory lookup primitives, in topological
   * order.
   */

  /**
   * Unlinks the entry 'child' from its existing parent directory 'dir'. Dual to
   * insert. This succeeds even if 'child' names a non-empty directory; we need
   * that for renameTo. 'child' must be a member of its parent directory,
   * however. Fails if the directory was read-only.
   */
  private void unlink(InMemoryDirectoryInfo dir, String child, Path errorPath)
      throws IOException {
    if (!dir.isWritable()) { throw Error.EACCES.exception(errorPath); }
    dir.removeChild(child);
  }

  /**
   * Inserts inode 'childInode' into the existing directory 'dir' under the
   * specified 'name'.  Dual to unlink.  Fails if the directory was read-only.
   */
  @CheckReturnValue
  private Error insert(InMemoryDirectoryInfo dir, String child,
                       InMemoryContentInfo childInode) {
    if (!dir.isWritable()) {
      return Error.EACCES;
    }
    dir.addChild(child, childInode);
    return null;
  }

  private void insert(InMemoryDirectoryInfo dir, String child,
      InMemoryContentInfo childInode, Path errorPath) throws IOException {
    Error error = insert(dir, child, childInode);
    if (error != null) {
      throw error.exception(errorPath);
    }
  }

  /**
   * Given an existing directory 'dir', looks up 'name' within it and returns
   * its inode. Assumes the file exists, unless 'create', in which case it will
   * try to create it. May fail with ENOTDIR, EACCES, ENOENT. Error messages
   * will be reported against file 'path'.
   */
  private InodeOrErrno directoryLookupErrno(InMemoryContentInfo dir,
                                            String name,
                                            boolean create,
                                            Path path) {
    if (!dir.isDirectory()) {
      return Error.ENOTDIR.asInodeOrErrno();
    }
    InMemoryDirectoryInfo imdi = (InMemoryDirectoryInfo) dir;
    if (!imdi.isExecutable()) {
      return Error.EACCES.asInodeOrErrno();
    }
    InMemoryContentInfo child = imdi.getChild(name);
    if (child == null) {
      if (!create)  {
        return Error.ENOENT.asInodeOrErrno();
      } else {
        child = newFile(clock, path);
        Error error = insert(imdi, name, child);
        if (error != null) {
          return error.asInodeOrErrno();
        }
      }
    }
    return InodeOrErrno.createInode(child);
  }

  protected FileInfo newFile(Clock clock, Path path) {
    return new InMemoryFileInfo(clock);
  }

  /**
   * Low-level path-to-inode lookup routine. Analogous to path_walk() in many UNIX kernels. Given
   * 'path', walks the directory tree from the root, resolving all symbolic links, and returns the
   * designated inode.
   *
   * <p>If 'create' is false, the inode must exist; otherwise, it will be created and added to its
   * parent directory, which must exist.
   *
   *
   * <p>May fail with ENOTDIR, ENOENT, EACCES, ELOOP.
   */
  private synchronized InodeOrErrno pathWalkErrno(Path path, boolean create) {
    Stack<String> stack = new Stack<>();
    for (Path p = path; !isRootDirectory(p); p = p.getParentDirectory()) {
      String name = baseNameOrWindowsDrive(p);
      stack.push(name);
    }

    InMemoryContentInfo inode = rootInode;
    int traversals = 0;

    while (!stack.isEmpty()) {
      traversals++;

      String name = stack.pop();

      // ENOENT on last segment with 'create' => create a new file.
      InodeOrErrno childOrError =
          directoryLookupErrno(inode, name, create && stack.isEmpty(), path);
      if (childOrError.hasError()) {
        return childOrError;
      }

      InMemoryContentInfo child = childOrError.inode();
      if (child.isSymbolicLink()) {
        PathFragment linkTarget = ((InMemoryLinkInfo) child).getNormalizedLinkContent();
        if (linkTarget.isAbsolute()) {
          inode = rootInode;
        }
        if (traversals > MAX_TRAVERSALS) {
          return Error.ELOOP.asInodeOrErrno();
        }
        List<String> segments = linkTarget.getSegments();
        for (int ii = segments.size() - 1; ii >= 0; --ii) {
          stack.push(segments.get(ii)); // Note this may include ".." segments.
        }
        // Push Windows drive if there is one
        if (linkTarget.isAbsolute()) {
          String driveStr = linkTarget.getDriveStr();
          if (driveStr.length() > 1) {
            stack.push(driveStr);
          }
        }
      } else {
        inode = child;
      }
    }
    return InodeOrErrno.createInode(inode);
  }

  /**
   * Given 'path', returns the existing directory inode it designates,
   * following symbolic links.
   *
   * <p>May fail with ENOTDIR, or any exception from pathWalk.
   */
  private InodeOrErrno getDirectoryErrno(Path path) {
    InodeOrErrno dirInfoOrError = pathWalkErrno(path, false);
    if (dirInfoOrError.hasError()) {
      return dirInfoOrError;
    }
    InMemoryContentInfo dirInfo = dirInfoOrError.inode();
    if (!dirInfo.isDirectory()) {
      return Error.ENOTDIR.asInodeOrErrno();
    } else {
      return dirInfoOrError;
    }
  }

  /**
   * Given 'path', returns the existing directory inode it designates,
   * following symbolic links.
   *
   * <p>May fail with ENOTDIR, or any exception from pathWalk.
   */
  private InMemoryDirectoryInfo getDirectory(Path path) throws IOException {
    return (InMemoryDirectoryInfo) getDirectoryErrno(path).valueOrThrow(path);
  }

  /**
   * Helper method for stat and inodeStat: return the path's (no symlink-followed) stat.
   */
  private synchronized InodeOrErrno noFollowStatErrno(Path path) {
    InodeOrErrno dirInfoOrError = getDirectoryErrno(path.getParentDirectory());
    if (dirInfoOrError.hasError()) {
      return dirInfoOrError;
    }
    return directoryLookupErrno(dirInfoOrError.inode(), baseNameOrWindowsDrive(path),
        /*create=*/ false, path);
  }

  /**
   * Given 'path', returns the existing inode it designates, optionally
   * following symbolic links.  Analogous to UNIX stat(2)/lstat(2), except that
   * it returns a mutable inode we can modify directly.
   */
  @Override
  public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    return inodeStatErrno(path, followSymlinks).valueOrThrow(path);
  }

  @Override
  @Nullable
  public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      InodeOrErrno inodeOrErrno = inodeStatErrno(path, followSymlinks);
    if (inodeOrErrno.hasError()) {
      Error errorCode = inodeOrErrno.error();
      if (errorCode == Error.ENOENT || errorCode == Error.ENOTDIR) {
        return null;
      }
      throw errorCode.exception(path);
    } else {
      return inodeOrErrno.inode();
    }
  }

  @Override
  protected FileStatus statNullable(Path path, boolean followSymlinks) {
    InodeOrErrno inodeOrErrno = inodeStatErrno(path, followSymlinks);
    return inodeOrErrno.hasError() ? null : inodeOrErrno.inode();
  }

  /**
   * Version of stat that returns an InodeOrErrno of the input path.
   */
  @CheckReturnValue
  protected InodeOrErrno inodeStatErrno(Path path, boolean followSymlinks) {
    if (followSymlinks) {
      return pathWalkErrno(path, false);
    } else {
      return isRootDirectory(path)
          ? InodeOrErrno.createInode(rootInode)
          : noFollowStatErrno(path);
    }
  }

  private InMemoryContentInfo inodeStat(Path path, boolean followSymlinks) throws IOException {
    return inodeStatErrno(path, followSymlinks).valueOrThrow(path);
  }

  /****************************************************************************
   *  FileSystem methods
   */

  /**
   * This is a helper routing for {@link #resolveSymbolicLinks(Path)}, i.e.
   * the "user-mode" routing for canonicalising paths. It is analogous to the
   * code in glibc's realpath(3).
   *
   * <p>Just like realpath, resolveSymbolicLinks requires a quadratic number of
   * directory lookups: n path segments are statted, and each stat requires a
   * linear amount of work in the "kernel" routine.
   */
  @Override
  protected PathFragment resolveOneLink(Path path) throws IOException {
    // Beware, this seemingly simple code belies the complex specification of
    // FileSystem.resolveOneLink().
    InMemoryContentInfo status = inodeStat(path, false);
    return status.isSymbolicLink() ? ((InMemoryLinkInfo) status).getLinkContent() : null;
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    return statNullable(path, followSymlinks) != null;
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    return status.isReadable();
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = inodeStat(path, true);
      status.setReadable(readable);
    }
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    return status.isWritable();
  }

  @Override
  public void setWritable(Path path, boolean writable) throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = inodeStat(path, true);
      status.setWritable(writable);
    }
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, true);
    return status.isExecutable();
  }

  @Override
  protected void setExecutable(Path path, boolean executable)
      throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = inodeStat(path, true);
      status.setExecutable(executable);
    }
  }

  @Override
  public boolean supportsModifications(Path path) {
    return true;
  }

  @Override
  public boolean supportsSymbolicLinksNatively(Path path) {
    return true;
  }

  @Override
  public boolean supportsHardLinksNatively(Path path) {
    return true;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return OS.getCurrent() != OS.WINDOWS;
  }

  @Override
  public boolean createDirectory(Path path) throws IOException {
    if (isRootDirectory(path)) {
      throw Error.EACCES.exception(path);
    }

    InMemoryDirectoryInfo parent;
    synchronized (this) {
      parent = getDirectory(path.getParentDirectory());
      InMemoryContentInfo child = parent.getChild(baseNameOrWindowsDrive(path));
      if (child != null) { // already exists
        if (child.isDirectory()) {
          return false;
        } else {
          throw Error.EEXIST.exception(path);
        }
      }

      InMemoryDirectoryInfo newDir = new InMemoryDirectoryInfo(clock);
      newDir.addChild(".", newDir);
      newDir.addChild("..", parent);
      insert(parent, baseNameOrWindowsDrive(path), newDir, path);

      return true;
    }
  }

  @Override
  public synchronized void createDirectoryAndParents(Path path) throws IOException {
    List<Path> subdirs = new ArrayList<>();
    for (; !isRootDirectory(path); path = path.getParentDirectory()) {
      if (path.isDirectory()) {
        break;
      } else if (path.exists()) {
        throw new IOException("Not a directory: " + path);
      }
      subdirs.add(path);
    }
    for (Path subdir : Lists.reverse(subdirs)) {
      subdir.createDirectory();
    }
  }

  @Override
  protected void createSymbolicLink(Path path, PathFragment targetFragment)
      throws IOException {
    if (isRootDirectory(path)) {
      throw Error.EACCES.exception(path);
    }

    synchronized (this) {
      InMemoryDirectoryInfo parent = getDirectory(path.getParentDirectory());
      if (parent.getChild(baseNameOrWindowsDrive(path)) != null) {
        throw Error.EEXIST.exception(path);
      }
      insert(
          parent, baseNameOrWindowsDrive(path), new InMemoryLinkInfo(clock, targetFragment), path);
    }
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    InMemoryContentInfo status = inodeStat(path, false);
    if (status.isSymbolicLink()) {
      Preconditions.checkState(status instanceof InMemoryLinkInfo);
      return ((InMemoryLinkInfo) status).getLinkContent();
    } else {
        throw new NotASymlinkException(path);
    }
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks)
      throws IOException {
    return stat(path, followSymlinks).getSize();
  }

  @Override
  protected Collection<String> getDirectoryEntries(Path path) throws IOException {
    synchronized (this) {
      InMemoryDirectoryInfo dirInfo = getDirectory(path);
      FileStatus status = stat(path, false);
      Preconditions.checkState(status instanceof InMemoryContentInfo);
      if (!((InMemoryContentInfo) status).isReadable()) {
        throw new IOException("Directory is not readable");
      }

      Collection<String> allChildren = dirInfo.getAllChildren();
      List<String> result = new ArrayList<>(allChildren.size());
      for (String child : allChildren) {
        if (!(child.equals(".") || child.equals(".."))) {
          result.add(child);
        }
      }
      return result;
    }
  }

  @Override
  public boolean delete(Path path) throws IOException {
    if (isRootDirectory(path)) {
      throw Error.EBUSY.exception(path);
    }
    if (!exists(path, false)) { return false; }

    synchronized (this) {
      InMemoryDirectoryInfo parent = getDirectory(path.getParentDirectory());
      InMemoryContentInfo child = parent.getChild(baseNameOrWindowsDrive(path));
      if (child.isDirectory() && child.getSize() > 2) {
        throw Error.ENOTEMPTY.exception(path);
      }
      unlink(parent, baseNameOrWindowsDrive(path), path);
      return true;
    }
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks)
      throws IOException {
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  public void setLastModifiedTime(Path path, long newTime) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = inodeStat(path, true);
      status.setLastModifiedTime(newTime == -1L ? clock.currentTimeMillis() : newTime);
    }
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = inodeStat(path, true);
      if (status.isDirectory()) {
        throw Error.EISDIR.exception(path);
      }
      if (!path.isReadable()) {
        throw Error.EACCES.exception(path);
      }
      Preconditions.checkState(status instanceof FileInfo);
      return ((FileInfo) status).getInputStream();
    }
  }

  @Override
  public byte[] getxattr(Path path, String name, boolean followSymlinks) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = inodeStat(path, followSymlinks);
      if (status.isDirectory()) {
        throw Error.EISDIR.exception(path);
      }
      if (!path.isReadable()) {
        throw Error.EACCES.exception(path);
      }
      Preconditions.checkState(status instanceof FileInfo);
      return ((FileInfo) status).getxattr(name);
    }
  }

  @Override
  protected byte[] getFastDigest(Path path) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = inodeStat(path, true);
      if (status.isDirectory()) {
        throw Error.EISDIR.exception(path);
      }
      if (!path.isReadable()) {
        throw Error.EACCES.exception(path);
      }
      Preconditions.checkState(status instanceof FileInfo);
      return ((FileInfo) status).getFastDigest();
    }
  }

  /** Creates a new file at the given path and returns its inode. */
  protected InMemoryContentInfo getOrCreateWritableInode(Path path) throws IOException {
    // open(WR_ONLY) of a dangling link writes through the link.  That means
    // that the usual path lookup operations have to behave differently when
    // resolving a path with the intent to create it: instead of failing with
    // ENOENT they have to return an open file.  This is exactly how UNIX
    // kernels do it, which is what we're trying to emulate.
    InMemoryContentInfo child = pathWalkErrno(path, /*create=*/true).valueOrThrow(path);
    Preconditions.checkNotNull(child);
    if (child.isDirectory()) {
      throw Error.EISDIR.exception(path);
    } else { // existing or newly-created file
      if (!child.isWritable()) { throw Error.EACCES.exception(path); }
      return child;
    }
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append)
      throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = getOrCreateWritableInode(path);
      return ((FileInfo) status).getOutputStream(append);
    }
  }

  @Override
  public void renameTo(Path sourcePath, Path targetPath)
      throws IOException {
    if (isRootDirectory(sourcePath)) {
      throw Error.EACCES.exception(sourcePath);
    }
    if (isRootDirectory(targetPath)) {
      throw Error.EACCES.exception(targetPath);
    }
    synchronized (this) {
      InMemoryDirectoryInfo sourceParent = getDirectory(sourcePath.getParentDirectory());
      InMemoryDirectoryInfo targetParent = getDirectory(targetPath.getParentDirectory());

      InMemoryContentInfo sourceInode = sourceParent.getChild(baseNameOrWindowsDrive(sourcePath));
      if (sourceInode == null) {
        throw Error.ENOENT.exception(sourcePath);
      }
      InMemoryContentInfo targetInode = targetParent.getChild(baseNameOrWindowsDrive(targetPath));

      unlink(sourceParent, baseNameOrWindowsDrive(sourcePath), sourcePath);
      try {
        // TODO(bazel-team): (2009) test with symbolic links.

        // Precondition checks:
        if (targetInode != null) { // already exists
          if (targetInode.isDirectory()) {
            if (!sourceInode.isDirectory()) {
              throw new IOException(sourcePath + " -> " + targetPath + " (" + Error.EISDIR + ")");
            }
            if (targetInode.getSize() > 2) {
              throw Error.ENOTEMPTY.exception(targetPath);
            }
          } else if (sourceInode.isDirectory()) {
            throw new IOException(sourcePath + " -> " + targetPath + " (" + Error.ENOTDIR + ")");
          }
          unlink(targetParent, baseNameOrWindowsDrive(targetPath), targetPath);
        }
        sourceInode.movedTo(targetPath);
        insert(targetParent, baseNameOrWindowsDrive(targetPath), sourceInode, targetPath);
        return;

      } catch (IOException e) {
        sourceInode.movedTo(sourcePath);
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
  protected void createFSDependentHardLink(Path linkPath, Path originalPath)
      throws IOException {

    // Same check used when creating a symbolic link
    if (isRootDirectory(originalPath)) {
      throw Error.EACCES.exception(originalPath);
    }

    synchronized (this) {
      InMemoryDirectoryInfo linkParent = getDirectory(linkPath.getParentDirectory());
      // Same check used when creating a symbolic link
      if (linkParent.getChild(baseNameOrWindowsDrive(linkPath)) != null) {
        throw Error.EEXIST.exception(linkPath);
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
  private boolean isRootDirectory(@Nullable Path path) {
    return path == null || path.getPathString().equals("/");
  }

  /**
   * Returns either the base name of the path, or the drive (if referring to a Windows drive).
   *
   * <p>This allows the file system to treat windows drives much like directories.
   */
  private static String baseNameOrWindowsDrive(Path path) {
    String name = path.getBaseName();
    return !name.isEmpty() ? name : path.getDriveStr();
  }

  /**
   * A class representing either an {@link Error} or an {@link InMemoryContentInfo}.
   */
  @AutoValue
  protected abstract static class InodeOrErrno {
    static InodeOrErrno createInode(InMemoryContentInfo info) {
      return new AutoValue_InMemoryFileSystem_InodeOrErrno(Preconditions.checkNotNull(info), null);
    }

    static InodeOrErrno createError(Error error) {
      return new AutoValue_InMemoryFileSystem_InodeOrErrno(null, Preconditions.checkNotNull(error));
    }

    @Nullable
    public abstract InMemoryContentInfo inode();

    @Nullable
    public abstract Error error();

    public boolean hasError() {
      return error() != null;
    }

    public InMemoryContentInfo valueOrThrow(Path path) throws IOException {
      if (hasError()) {
        throw error().exception(path);
      }
      return inode();
    }
  }
}
