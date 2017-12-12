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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileAccessException;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayInputStream;
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
public class InMemoryFileSystem extends FileSystem {

  private final PathFragment scopeRoot;
  private final Clock clock;

  // The root inode (a directory).
  private final InMemoryDirectoryInfo rootInode;

  // Maximum number of traversals before ELOOP is thrown.
  private static final int MAX_TRAVERSALS = 256;

  /**
   * Creates a new InMemoryFileSystem with scope checking disabled (all paths are considered to be
   * within scope) and a default clock.
   */
  public InMemoryFileSystem() {
    this(new JavaClock());
  }

  /**
   * Creates a new InMemoryFileSystem with scope checking disabled (all
   * paths are considered to be within scope).
   */
  public InMemoryFileSystem(Clock clock) {
    this(clock, (PathFragment) null);
  }

  /**
   * Creates a new InMemoryFileSystem with scope checking disabled (all paths are considered to be
   * within scope).
   */
  public InMemoryFileSystem(Clock clock, HashFunction hashFunction) {
    super(hashFunction);
    this.clock = clock;
    this.rootInode = newRootInode(clock);
    this.scopeRoot = null;
  }

  /**
   * Creates a new InMemoryFileSystem with scope checking bound to scopeRoot, i.e. any path that's
   * not below scopeRoot is considered to be out of scope.
   */
  public InMemoryFileSystem(Clock clock, PathFragment scopeRoot) {
    this.scopeRoot = scopeRoot;
    this.clock = clock;
    this.rootInode = newRootInode(clock);
  }

  private static InMemoryDirectoryInfo newRootInode(Clock clock) {
    InMemoryDirectoryInfo rootInode = new InMemoryDirectoryInfo(clock);
    rootInode.addChild(".", rootInode);
    rootInode.addChild("..", rootInode);
    return rootInode;
  }

  /**
   * Returns true if the given path is within this file system's scope, false otherwise.
   *
   * @param parentDepth the number of segments in the path's parent directory (only meaningful for
   *     paths that begin with ".."). The parent directory itself is assumed to be in scope.
   * @param normalizedPath input path, expected to be normalized such that all ".." and "." segments
   *     are removed (with the exception of a possible prefix sequence of contiguous ".." segments)
   */
  private boolean inScope(int parentDepth, PathFragment normalizedPath) {
    if (scopeRoot == null) {
      return true;
    } else if (normalizedPath.isAbsolute()) {
      return normalizedPath.startsWith(scopeRoot);
    } else {
      // Efficiency note: we're not accounting for "/scope/root/../root" paths here, i.e. paths
      // that appear to go out of scope but ultimately stay within scope. This may result in
      // unnecessary re-delegation back into the same FS. we're choosing to forgo that
      // optimization under the assumption that such scenarios are rare and unimportant to
      // overall performance. We can always enhance this if needed.
      return parentDepth - leadingParentReferences(normalizedPath) >= scopeRoot.segmentCount();
    }
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
  private void insert(InMemoryDirectoryInfo dir, String child,
                      InMemoryContentInfo childInode, Path errorPath)
      throws IOException {
    if (!dir.isWritable()) { throw Error.EACCES.exception(errorPath); }
    dir.addChild(child, childInode);
  }

  /**
   * Given an existing directory 'dir', looks up 'name' within it and returns
   * its inode. Assumes the file exists, unless 'create', in which case it will
   * try to create it. May fail with ENOTDIR, EACCES, ENOENT. Error messages
   * will be reported against file 'path'.
   */
  private InMemoryContentInfo directoryLookup(InMemoryContentInfo dir,
                                              String name,
                                              boolean create,
                                              Path path) throws IOException {
    if (!dir.isDirectory()) { throw Error.ENOTDIR.exception(path); }
    InMemoryDirectoryInfo imdi = (InMemoryDirectoryInfo) dir;
    if (!imdi.isExecutable()) { throw Error.EACCES.exception(path); }
    InMemoryContentInfo child = imdi.getChild(name);
    if (child == null) {
      if (!create)  {
        throw Error.ENOENT.exception(path);
      } else {
        child = new InMemoryFileInfo(clock);
        insert(imdi, name, child, path);
      }
    }
    return child;
  }

  /**
   * Low-level path-to-inode lookup routine. Analogous to path_walk() in many UNIX kernels. Given
   * 'path', walks the directory tree from the root, resolving all symbolic links, and returns the
   * designated inode.
   *
   * <p>If 'create' is false, the inode must exist; otherwise, it will be created and added to its
   * parent directory, which must exist.
   *
   * <p>Iff the given path escapes this file system's scope, a Error.ENOENT exception is thrown.
   *
   * <p>May fail with ENOTDIR, ENOENT, EACCES, ELOOP.
   */
  private synchronized InMemoryContentInfo pathWalk(Path path, boolean create) throws IOException {
    // Implementation note: This is where we check for out-of-scope symlinks and
    // trigger re-delegation to another file system accordingly. This code handles
    // both absolute and relative symlinks. Some assumptions we make: First, only
    // symlink targets as read from getNormalizedLinkContent() can escape our scope.
    // This is because Path objects are all canonicalized (see {@link Path#getRelative},
    // etc.) and symlink target segments that get added to the stack are in-scope by
    // definition. Second, symlink targets with relative segments must have the form
    // [".."]*[standard segment]+, i.e. only the ".." non-standard segment is allowed
    // and it may only appear as part of a contiguous prefix sequence.

    Stack<String> stack = new Stack<>();
    PathFragment rootPathFragment = getRootDirectory().asFragment();
    for (Path p = path; !p.asFragment().equals(rootPathFragment); p = p.getParentDirectory()) {
      stack.push(p.getBaseName());
    }

    InMemoryContentInfo inode = rootInode;
    int parentDepth = -1;
    int traversals = 0;

    while (!stack.isEmpty()) {
      traversals++;

      String name = stack.pop();
      parentDepth += name.equals("..") ? -1 : 1;

      // ENOENT on last segment with 'create' => create a new file.
      InMemoryContentInfo child = directoryLookup(inode, name, create && stack.isEmpty(), path);
      if (child.isSymbolicLink()) {
        PathFragment linkTarget = ((InMemoryLinkInfo) child).getNormalizedLinkContent();
        if (!inScope(parentDepth, linkTarget)) {
          throw Error.ENOENT.exception(path);
        }
        if (linkTarget.isAbsolute()) {
          inode = rootInode;
          parentDepth = -1;
        }
        if (traversals > MAX_TRAVERSALS) {
          throw Error.ELOOP.exception(path);
        }
        for (int ii = linkTarget.segmentCount() - 1; ii >= 0; --ii) {
          stack.push(linkTarget.getSegment(ii)); // Note this may include ".." segments.
        }
      } else {
        inode = child;
      }
    }
    return inode;
  }

  /**
   * Given 'path', returns the existing directory inode it designates,
   * following symbolic links.
   *
   * <p>May fail with ENOTDIR, or any exception from pathWalk.
   */
  private InMemoryDirectoryInfo getDirectory(Path path) throws IOException {
    InMemoryContentInfo dirInfo = pathWalk(path, false);
    if (!dirInfo.isDirectory()) {
      throw Error.ENOTDIR.exception(path);
    } else {
      return (InMemoryDirectoryInfo) dirInfo;
    }
  }

  /**
   * Helper method for stat, scopeLimitedStat: lock the internal state and return the
   * path's (no symlink-followed) stat if the path's parent directory is within scope,
   * else return an "out of scope" reference to the path's parent directory (which will
   * presumably be re-delegated to another FS).
   */
  private synchronized InMemoryContentInfo getNoFollowStatOrOutOfScopeParent(Path path)
      throws IOException  {
    InMemoryDirectoryInfo dirInfo = getDirectory(path.getParentDirectory());
    return directoryLookup(dirInfo, path.getBaseName(), /*create=*/ false, path);
  }

  /**
   * Given 'path', returns the existing inode it designates, optionally
   * following symbolic links.  Analogous to UNIX stat(2)/lstat(2), except that
   * it returns a mutable inode we can modify directly.
   */
  @Override
  public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    if (followSymlinks) {
      return scopeLimitedStat(path, true);
    } else {
      if (path.equals(getRootDirectory())) {
        return rootInode;
      } else {
        return getNoFollowStatOrOutOfScopeParent(path);
      }
    }
  }

  @Override
  @Nullable
  public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
    try {
      return stat(path, followSymlinks);
    } catch (IOException e) {
      if (e instanceof Error.WithError) {
        Error errorCode = ((Error.WithError) e).getError();
        if  (errorCode == Error.ENOENT || errorCode == Error.ENOTDIR) {
          return null;
        }
      }
      throw e;
    }
  }

  /**
   * Version of stat that returns an inode if the input path stays entirely within this file
   * system's scope, otherwise throws.
   */
  private InMemoryContentInfo scopeLimitedStat(Path path, boolean followSymlinks)
      throws IOException {
    if (followSymlinks) {
      return pathWalk(path, false);
    } else {
      if (path.equals(getRootDirectory())) {
        return rootInode;
      } else {
        return getNoFollowStatOrOutOfScopeParent(path);
      }
    }
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
    InMemoryContentInfo status = scopeLimitedStat(path, false);
    return status.isSymbolicLink() ? ((InMemoryLinkInfo) status).getLinkContent() : null;
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks).isDirectory();
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks).isFile();
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks).isSpecialFile();
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    try {
      return stat(path, false).isSymbolicLink();
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    try {
      stat(path, followSymlinks);
      return true;
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, true);
    return status.isReadable();
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = scopeLimitedStat(path, true);
      status.setReadable(readable);
    }
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, true);
    return status.isWritable();
  }

  @Override
  public void setWritable(Path path, boolean writable) throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = scopeLimitedStat(path, true);
      status.setWritable(writable);
    }
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, true);
    return status.isExecutable();
  }

  @Override
  protected void setExecutable(Path path, boolean executable)
      throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = scopeLimitedStat(path, true);
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
    if (path.equals(getRootDirectory())) {
      throw Error.EACCES.exception(path);
    }

    InMemoryDirectoryInfo parent;
    synchronized (this) {
      parent = getDirectory(path.getParentDirectory());
      InMemoryContentInfo child = parent.getChild(path.getBaseName());
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
      insert(parent, path.getBaseName(), newDir, path);

      return true;
    }
  }

  @Override
  protected void createSymbolicLink(Path path, PathFragment targetFragment)
      throws IOException {
    if (path.equals(getRootDirectory())) {
      throw Error.EACCES.exception(path);
    }

    synchronized (this) {
      InMemoryDirectoryInfo parent = getDirectory(path.getParentDirectory());
      if (parent.getChild(path.getBaseName()) != null) {
        throw Error.EEXIST.exception(path);
      }
      insert(parent, path.getBaseName(), new InMemoryLinkInfo(clock, targetFragment), path);
    }
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, false);
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
    if (path.equals(getRootDirectory())) {
      throw Error.EBUSY.exception(path);
    }
    if (!exists(path, false)) { return false; }

    synchronized (this) {
      InMemoryDirectoryInfo parent = getDirectory(path.getParentDirectory());
      InMemoryContentInfo child = parent.getChild(path.getBaseName());
      if (child.isDirectory() && child.getSize() > 2) {
        throw Error.ENOTEMPTY.exception(path);
      }
      unlink(parent, path.getBaseName(), path);
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
      InMemoryContentInfo status = scopeLimitedStat(path, true);
      status.setLastModifiedTime(newTime == -1L ? clock.currentTimeMillis() : newTime);
    }
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    synchronized (this) {
      InMemoryContentInfo status = scopeLimitedStat(path, true);
      if (status.isDirectory()) {
        throw Error.EISDIR.exception(path);
      }
      if (!path.isReadable()) {
        throw Error.EACCES.exception(path);
      }
      Preconditions.checkState(status instanceof FileInfo);
      return new ByteArrayInputStream(((FileInfo) status).readContent());
    }
  }

  /** Creates a new file at the given path and returns its inode. */
  private InMemoryContentInfo getOrCreateWritableInode(Path path) throws IOException {
    // open(WR_ONLY) of a dangling link writes through the link.  That means
    // that the usual path lookup operations have to behave differently when
    // resolving a path with the intent to create it: instead of failing with
    // ENOENT they have to return an open file.  This is exactly how UNIX
    // kernels do it, which is what we're trying to emulate.
    InMemoryContentInfo child = pathWalk(path, /*create=*/true);
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
    if (sourcePath.equals(getRootDirectory())) {
      throw Error.EACCES.exception(sourcePath);
    }
    if (targetPath.equals(getRootDirectory())) {
      throw Error.EACCES.exception(targetPath);
    }
    synchronized (this) {
      InMemoryDirectoryInfo sourceParent = getDirectory(sourcePath.getParentDirectory());
      InMemoryDirectoryInfo targetParent = getDirectory(targetPath.getParentDirectory());

      InMemoryContentInfo sourceInode = sourceParent.getChild(sourcePath.getBaseName());
      if (sourceInode == null) {
        throw Error.ENOENT.exception(sourcePath);
      }
      InMemoryContentInfo targetInode = targetParent.getChild(targetPath.getBaseName());

      unlink(sourceParent, sourcePath.getBaseName(), sourcePath);
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
          unlink(targetParent, targetPath.getBaseName(), targetPath);
        }
        sourceInode.movedTo(targetPath);
        insert(targetParent, targetPath.getBaseName(), sourceInode, targetPath);
        return;

      } catch (IOException e) {
        sourceInode.movedTo(sourcePath);
        insert(sourceParent, sourcePath.getBaseName(), sourceInode, sourcePath); // restore source
        throw e;
      }
    }
  }

  @Override
  protected void createFSDependentHardLink(Path linkPath, Path originalPath)
      throws IOException {

    // Same check used when creating a symbolic link
    if (originalPath.equals(getRootDirectory())) {
      throw Error.EACCES.exception(originalPath);
    }

    synchronized (this) {
      InMemoryDirectoryInfo linkParent = getDirectory(linkPath.getParentDirectory());
      // Same check used when creating a symbolic link
      if (linkParent.getChild(linkPath.getBaseName()) != null) {
        throw Error.EEXIST.exception(linkPath);
      }
      insert(
          linkParent,
          linkPath.getBaseName(),
          getDirectory(originalPath.getParentDirectory()).getChild(originalPath.getBaseName()),
          linkPath);
    }
  }
}
