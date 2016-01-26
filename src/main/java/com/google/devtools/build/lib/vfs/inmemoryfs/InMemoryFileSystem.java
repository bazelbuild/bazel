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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.unix.FileAccessException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.JavaClock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.ScopeEscapableFileSystem;
import com.google.devtools.build.lib.vfs.Symlinks;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.Stack;

import javax.annotation.Nullable;

/**
 * This class provides a complete in-memory file system.
 *
 * <p>Naming convention: we use "path" for all {@link Path} variables, since these
 * represent *names* and we use "node" or "inode" for InMemoryContentInfo
 * variables, since these correspond to inodes in the UNIX file system.
 *
 * <p>The code is structured to be as similar to the implementation of UNIX "namei"
 * as is reasonably possibly. This provides a firm reference point for many
 * concepts and makes compatibility easier to achieve.
 *
 * <p>As a scope-escapable file system, this class supports re-delegation of symbolic links
 * that escape its root. This is done through the use of {@link OutOfScopeFileStatus}
 * and {@link OutOfScopeDirectoryStatus} objects, which may be returned by
 * getDirectory, pathWalk, and scopeLimitedStat. Any code that calls one of these
 * methods (either directly or indirectly) is obligated to check the possibility
 * that its info represents an out-of-scope path. Lack of such a check will result
 * in unchecked runtime exceptions upon any request for status data (as well as
 * possible logical errors).
 */
@ThreadSafe
public class InMemoryFileSystem extends ScopeEscapableFileSystem {

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
    this(clock, null);
  }

  /**
   * Creates a new InMemoryFileSystem with scope checking bound to
   * scopeRoot, i.e. any path that's not below scopeRoot is considered
   * to be out of scope.
   */
  protected InMemoryFileSystem(Clock clock, PathFragment scopeRoot) {
    super(scopeRoot);
    this.clock = clock;
    this.rootInode = new InMemoryDirectoryInfo(clock);
    rootInode.addChild(".", rootInode);
    rootInode.addChild("..", rootInode);
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

    private Error(String message) {
      this.message = message;
    }

    @Override
    public String toString() {
      return message;
    }

    /** Implemented by exceptions that contain the extra info of which Error caused them. */
    private static interface WithError {
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
        child = makeFileInfo(clock, path.asFragment());
        insert(imdi, name, child, path);
      }
    }
    return child;
  }

  /**
   * Low-level path-to-inode lookup routine. Analogous to path_walk() in many
   * UNIX kernels. Given 'path', walks the directory tree from the root,
   * resolving all symbolic links, and returns the designated inode.
   *
   * <p>If 'create' is false, the inode must exist; otherwise, it will be created
   * and added to its parent directory, which must exist.
   *
   * <p>Iff the given path escapes this file system's scope, the returned value
   * is an {@link OutOfScopeFileStatus} instance. Any code that calls this method
   * needs to check for that possibility (via {@link ScopeEscapableStatus#outOfScope}).
   *
   * <p>May fail with ENOTDIR, ENOENT, EACCES, ELOOP.
   */
  private synchronized InMemoryContentInfo pathWalk(Path path, boolean create)
      throws IOException {
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
    PathFragment rootPathFragment = rootPath.asFragment();
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
          return outOfScopeStatus(linkTarget, parentDepth, stack);
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
   * Helper routine for pathWalk: given a symlink target known to escape this file system's
   * scope (and that has the form [".."]*[standard segment]+), the number of segments
   * in the directory containing the symlink, and the remaining path segments following
   * the symlink in the original input to pathWalk, returns an OutofScopeFileStatus
   * initialized with an appropriate out-of-scope reformulation of pathWalk's original
   * input.
   */
  private OutOfScopeFileStatus outOfScopeStatus(PathFragment linkTarget, int parentDepth,
      Stack<String> descendantSegments) {

    PathFragment escapingPath;
    if (linkTarget.isAbsolute()) {
      escapingPath = linkTarget;
    } else {
      // Relative out-of-scope paths must look like "../../../a/b/c". Find the target's
      // parent path depth by subtracting one from parentDepth for each ".." reference.
      // Then use that to retrieve a prefix of the scope root, which is the target's
      // canonicalized parent path.
      int leadingParentRefs = leadingParentReferences(linkTarget);
      int baseDepth = parentDepth - leadingParentRefs;
      Preconditions.checkState(baseDepth < scopeRoot.segmentCount());
      escapingPath = baseDepth > 0
          ? scopeRoot.subFragment(0, baseDepth)
          : scopeRoot.subFragment(0, 0);
      // Now add in everything that comes after the ".." sequence.
      for (int i = leadingParentRefs; i < linkTarget.segmentCount(); i++) {
        escapingPath = escapingPath.getRelative(linkTarget.getSegment(i));
      }
    }

    // We've now converted the symlink to its target in canonicalized absolute path
    // form. Since the symlink wasn't necessarily the final segment in the original
    // input sent to pathWalk, now add in every segment that came after.
    while (!descendantSegments.empty()) {
      escapingPath = escapingPath.getRelative(descendantSegments.pop());
    }

    return new OutOfScopeFileStatus(escapingPath);
  }

  /**
   * Given 'path', returns the existing directory inode it designates,
   * following symbolic links.
   *
   * <p>May fail with ENOTDIR, or any exception from pathWalk.
   *
   * <p>Iff the given path escapes this file system's scope, this method skips
   * ENOTDIR checking and returns an OutOfScopeDirectoryStatus instance. Any
   * code that calls this method needs to check for that possibility
   * (via {@link ScopeEscapableStatus#outOfScope}).
   */
  private InMemoryDirectoryInfo getDirectory(Path path) throws IOException {
    InMemoryContentInfo dirInfo = pathWalk(path, false);
    if (dirInfo.outOfScope()) {
      return new OutOfScopeDirectoryStatus(dirInfo.getEscapingPath());
    } else if (!dirInfo.isDirectory()) {
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
    return dirInfo.outOfScope()
        ? dirInfo
        : directoryLookup(dirInfo, path.getBaseName(), /*create=*/false, path);
  }

  /**
   * Given 'path', returns the existing inode it designates, optionally
   * following symbolic links.  Analogous to UNIX stat(2)/lstat(2), except that
   * it returns a mutable inode we can modify directly.
   */
  @Override
  public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    if (followSymlinks) {
      InMemoryContentInfo status = scopeLimitedStat(path, true);
      return status.outOfScope()
          ? statWithDelegator(status.getEscapingPath(), true)
          : status;
    } else {
      if (path.equals(rootPath)) {
        return rootInode;
      } else {
        InMemoryContentInfo status = getNoFollowStatOrOutOfScopeParent(path);
        // If out of scope, status references the path's parent directory. Else it references the
        // path itself.
        return status.outOfScope()
            ? getDelegatedPath(status.getEscapingPath().getRelative(
                  path.getBaseName())).stat(Symlinks.NOFOLLOW)
            : status;
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
   * Version of stat that returns an inode if the input path stays entirely within
   * this file system's scope, otherwise an {@link OutOfScopeFileStatus}.
   *
   * <p>Any code that calls this method needs to check for either possibility via
   * {@link ScopeEscapableStatus#outOfScope}.
   */
  protected InMemoryContentInfo scopeLimitedStat(Path path, boolean followSymlinks)
      throws IOException {
    if (followSymlinks) {
      return pathWalk(path, false);
    } else {
      if (path.equals(rootPath)) {
        return rootInode;
      } else {
        InMemoryContentInfo status = getNoFollowStatOrOutOfScopeParent(path);
        // If out of scope, status references the path's parent directory. Else it references the
        // path itself.
        return status.outOfScope()
            ? new OutOfScopeFileStatus(status.getEscapingPath().getRelative(path.getBaseName()))
            : status;
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
    if (status.outOfScope()) {
      return resolveOneLinkWithDelegator(status.getEscapingPath());
    } else {
      return status.isSymbolicLink()
          ? ((InMemoryLinkInfo) status).getLinkContent()
          : null;
    }
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

  /**
   * Like {@link #exists}, but checks for existence within this filesystem's scope.
   */
  protected boolean scopeLimitedExists(Path path, boolean followSymlinks) {
    try {
      // Path#asFragment() always returns an absolute path, so inScope() is called with
      // parentDepth = 0.
      return inScope(0, path.asFragment()) && !scopeLimitedStat(path, followSymlinks).outOfScope();
    } catch (IOException e) {
      return false;
    }
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, true);
    return status.outOfScope()
        ? getDelegatedPath(status.getEscapingPath()).isReadable()
        : status.isReadable();
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = scopeLimitedStat(path, true);
      if (!status.outOfScope()) {
        status.setReadable(readable);
        return;
      }
    }
    // If we get here, we're out of scope.
    getDelegatedPath(status.getEscapingPath()).setReadable(readable);
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, true);
    return status.outOfScope()
        ? getDelegatedPath(status.getEscapingPath()).isWritable()
        : status.isWritable();
  }

  @Override
  protected void setWritable(Path path, boolean writable) throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = scopeLimitedStat(path, true);
      if (!status.outOfScope()) {
        status.setWritable(writable);
        return;
      }
    }
    // If we get here, we're out of scope.
    getDelegatedPath(status.getEscapingPath()).setWritable(writable);
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, true);
    return status.outOfScope()
        ? getDelegatedPath(status.getEscapingPath()).isExecutable()
        : status.isExecutable();
  }

  @Override
  protected void setExecutable(Path path, boolean executable)
      throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = scopeLimitedStat(path, true);
      if (!status.outOfScope()) {
        status.setExecutable(executable);
        return;
      }
    }
    // If we get here, we're out of scope.
    getDelegatedPath(status.getEscapingPath()).setExecutable(executable);
  }

  @Override
  public boolean supportsModifications() {
    return true;
  }

  @Override
  public boolean supportsSymbolicLinksNatively() {
    return true;
  }

  /**
   * Constructs a new inode.  Provided so that subclasses of InMemoryFileSystem
   * can inject subclasses of FileInfo properly.
   */
  protected FileInfo makeFileInfo(Clock clock, PathFragment frag) {
    return new InMemoryFileInfo(clock);
  }

  /**
   * Returns a new path constructed by appending the child's base name to the
   * escaped parent path. For example, assume our file system root is /foo
   * and /foo/link1 -> /bar. This method can be used on child = /foo/link1/link2/name
   * and parent = /bar/link2 to return /bar/link2/name, which is a semi-resolved
   * path bound to a different file system.
   */
  private Path getDelegatedPath(PathFragment escapedParent, Path child) {
    return getDelegatedPath(escapedParent.getRelative(child.getBaseName()));
  }

  @Override
  protected boolean createDirectory(Path path) throws IOException {
    if (path.equals(rootPath)) { throw Error.EACCES.exception(path); }

    InMemoryDirectoryInfo parent;
    synchronized (this) {
      parent = getDirectory(path.getParentDirectory());
      if (!parent.outOfScope()) {
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

    // If we get here, we're out of scope.
    return getDelegatedPath(parent.getEscapingPath(), path).createDirectory();
  }

  @Override
  protected void createSymbolicLink(Path path, PathFragment targetFragment)
      throws IOException {
    if (path.equals(rootPath)) { throw Error.EACCES.exception(path); }

    InMemoryDirectoryInfo parent;
    synchronized (this) {
      parent = getDirectory(path.getParentDirectory());
      if (!parent.outOfScope()) {
        if (parent.getChild(path.getBaseName()) != null) { throw Error.EEXIST.exception(path); }
        insert(parent, path.getBaseName(), new InMemoryLinkInfo(clock, targetFragment), path);
        return;
      }
    }

    // If we get here, we're out of scope.
    getDelegatedPath(parent.getEscapingPath(), path).createSymbolicLink(targetFragment);
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    InMemoryContentInfo status = scopeLimitedStat(path, false);
    if (status.outOfScope()) {
      return getDelegatedPath(status.getEscapingPath()).readSymbolicLink();
    } else if (status.isSymbolicLink()) {
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
  protected Collection<Path> getDirectoryEntries(Path path) throws IOException {
    InMemoryDirectoryInfo dirInfo;
    synchronized (this) {
      dirInfo = getDirectory(path);
      if (!dirInfo.outOfScope()) {
        FileStatus status = stat(path, false);
        Preconditions.checkState(status instanceof InMemoryContentInfo);
        if (!((InMemoryContentInfo) status).isReadable()) {
          throw new IOException("Directory is not readable");
        }

        Set<String> allChildren = dirInfo.getAllChildren();
        List<Path> result = new ArrayList<>(allChildren.size());
        for (String child : allChildren) {
          if (!(child.equals(".") || child.equals(".."))) {
            result.add(path.getChild(child));
          }
        }
        return result;
      }
    }

    // If we get here, we're out of scope.
    return getDelegatedPath(dirInfo.getEscapingPath()).getDirectoryEntries();
  }

  @Override
  protected boolean delete(Path path) throws IOException {
    if (path.equals(rootPath)) { throw Error.EBUSY.exception(path); }
    if (!exists(path, false)) { return false; }

    InMemoryDirectoryInfo parent;
    synchronized (this) {
      parent = getDirectory(path.getParentDirectory());
      if (!parent.outOfScope()) {
        InMemoryContentInfo child = parent.getChild(path.getBaseName());
        if (child.isDirectory() && child.getSize() > 2) { throw Error.ENOTEMPTY.exception(path); }
        unlink(parent, path.getBaseName(), path);
        return true;
      }
    }

    // If we get here, we're out of scope.
    return getDelegatedPath(parent.getEscapingPath(), path).delete();
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks)
      throws IOException {
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  protected void setLastModifiedTime(Path path, long newTime) throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = scopeLimitedStat(path, true);
      if (!status.outOfScope()) {
        status.setLastModifiedTime(newTime == -1L
                                   ? clock.currentTimeMillis()
                                   : newTime);
        return;
      }
    }

    // If we get here, we're out of scope.
    getDelegatedPath(status.getEscapingPath()).setLastModifiedTime(newTime);
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = scopeLimitedStat(path, true);
      if (!status.outOfScope()) {
        if (status.isDirectory()) { throw Error.EISDIR.exception(path); }
        if (!path.isReadable()) { throw Error.EACCES.exception(path); }
        Preconditions.checkState(status instanceof FileInfo);
        return new ByteArrayInputStream(((FileInfo) status).readContent());
      }
    }

    // If we get here, we're out of scope.
    return getDelegatedPath(status.getEscapingPath()).getInputStream();
  }

  /**
   * Creates a new file at the given path and returns its inode. If the path
   * escapes this file system's scope, trivially returns an "out of scope" status.
   * Calling code should check for both possibilities via
   * {@link ScopeEscapableStatus#outOfScope}.
   */
  protected InMemoryContentInfo getOrCreateWritableInode(Path path)
      throws IOException {
    // open(WR_ONLY) of a dangling link writes through the link.  That means
    // that the usual path lookup operations have to behave differently when
    // resolving a path with the intent to create it: instead of failing with
    // ENOENT they have to return an open file.  This is exactly how UNIX
    // kernels do it, which is what we're trying to emulate.
    InMemoryContentInfo child = pathWalk(path, /*create=*/true);
    Preconditions.checkNotNull(child);
    if (child.outOfScope()) {
      return child;
    } else if (child.isDirectory()) {
      throw Error.EISDIR.exception(path);
    } else { // existing or newly-created file
      if (!child.isWritable()) { throw Error.EACCES.exception(path); }
      return child;
    }
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append)
      throws IOException {
    InMemoryContentInfo status;
    synchronized (this) {
      status = getOrCreateWritableInode(path);
      if (!status.outOfScope()) {
        return ((FileInfo) getOrCreateWritableInode(path)).getOutputStream(append);
      }
    }
    // If we get here, we're out of scope.
    return getDelegatedPath(status.getEscapingPath()).getOutputStream(append);
  }

  @Override
  protected void renameTo(Path sourcePath, Path targetPath)
      throws IOException {
    if (sourcePath.equals(rootPath)) { throw Error.EACCES.exception(sourcePath); }
    if (targetPath.equals(rootPath)) { throw Error.EACCES.exception(targetPath); }

    InMemoryDirectoryInfo sourceParent;
    InMemoryDirectoryInfo targetParent;

    synchronized (this) {
      sourceParent = getDirectory(sourcePath.getParentDirectory());
      targetParent = getDirectory(targetPath.getParentDirectory());

      // Handle the rename if both paths are within our scope.
      if (!sourceParent.outOfScope() && !targetParent.outOfScope()) {
        InMemoryContentInfo sourceInode = sourceParent.getChild(sourcePath.getBaseName());
        if (sourceInode == null) { throw Error.ENOENT.exception(sourcePath); }
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

    // If we get here, either one or both paths is out of scope.
    if (sourceParent.outOfScope() && targetParent.outOfScope()) {
      Path delegatedSource = getDelegatedPath(sourceParent.getEscapingPath(), sourcePath);
      Path delegatedTarget = getDelegatedPath(targetParent.getEscapingPath(), targetPath);
      delegatedSource.renameTo(delegatedTarget);
    } else {
      // We don't support cross-file system renaming.
      throw Error.EACCES.exception(targetPath);
    }
  }
}
