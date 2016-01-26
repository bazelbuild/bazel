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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.unix.ErrnoFileStatus;
import com.google.devtools.build.lib.unix.FilesystemUtils;
import com.google.devtools.build.lib.unix.FilesystemUtils.Dirents;
import com.google.devtools.build.lib.unix.FilesystemUtils.ReadTypes;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * This class implements the FileSystem interface using direct calls to the
 * UNIX filesystem.
 */
// Not final only for testing.
@ThreadSafe
public class UnixFileSystem extends AbstractFileSystemWithCustomStat {
  /**
   * What to do with requests to create symbolic links.
   *
   * Currently supports one value: SYMLINK, which simply calls symlink() . It obviously does not
   * work on Windows.
   */
  public enum SymlinkStrategy {
    /**
     * Use symlink(). Does not work on Windows, obviously.
     */
    SYMLINK,

    /**
     * Write a log message for symlinks that won't be compatible with how we are planning to pretend
     * that they exist on Windows.
     */
    WINDOWS_COMPATIBLE,
  }

  private final SymlinkStrategy symlinkStrategy;
  private final String symlinkLogFile;

  /**
   * Directories where Bazel tries to hardlink files from instead of copying them.
   *
   * <p>These must be writable to the user.
   */
  private ImmutableList<Path> rootsWithAllowedHardlinks;

  public UnixFileSystem() {
    SymlinkStrategy symlinkStrategy = SymlinkStrategy.SYMLINK;
    String strategyString = System.getProperty("io.bazel.SymlinkStrategy");
    symlinkLogFile = System.getProperty("io.bazel.SymlinkLogFile");
    if (strategyString != null) {
      try {
        symlinkStrategy = SymlinkStrategy.valueOf(strategyString.toUpperCase());
      } catch (IllegalArgumentException e) {
        // We just go with the default, this is just an experimental option so it's fine.
      }

      if (symlinkLogFile != null) {
        writeLogMessage("Logging started");
      }
    }

    this.symlinkStrategy = symlinkStrategy;
    rootsWithAllowedHardlinks = ImmutableList.of();
  }

  // This method is a little ugly, but it's only for testing for now.
  public void setRootsWithAllowedHardlinks(Iterable<Path> roots) {
    this.rootsWithAllowedHardlinks = ImmutableList.copyOf(roots);
  }

  /**
   * Eager implementation of FileStatus for file systems that have an atomic
   * stat(2) syscall. A proxy for {@link com.google.devtools.build.lib.unix.FileStatus}.
   * Note that isFile and getLastModifiedTime have slightly different meanings
   * between UNIX and VFS.
   */
  @VisibleForTesting
  protected static class UnixFileStatus implements FileStatus {

    private final com.google.devtools.build.lib.unix.FileStatus status;

    UnixFileStatus(com.google.devtools.build.lib.unix.FileStatus status) {
      this.status = status;
    }

    @Override
    public boolean isFile() { return !isDirectory() && !isSymbolicLink(); }

    @Override
    public boolean isDirectory() { return status.isDirectory(); }

    @Override
    public boolean isSymbolicLink() { return status.isSymbolicLink(); }

    @Override
    public boolean isSpecialFile() { return isFile() && !status.isRegularFile(); }

    @Override
    public long getSize() { return status.getSize(); }

    @Override
    public long getLastModifiedTime() {
      return (status.getLastModifiedTime() * 1000)
          + (status.getFractionalLastModifiedTime() / 1000000);
    }

    @Override
    public long getLastChangeTime() {
      return (status.getLastChangeTime() * 1000)
          + (status.getFractionalLastChangeTime() / 1000000);
    }

    @Override
    public long getNodeId() {
      // Note that we may want to include more information in this id number going forward,
      // especially the device number.
      return status.getInodeNumber();
    }

    int getPermissions() { return status.getPermissions(); }

    @Override
    public String toString() { return status.toString(); }
  }

  @Override
  protected Collection<Path> getDirectoryEntries(Path path) throws IOException {
    String name = path.getPathString();
    String[] entries;
    long startTime = Profiler.nanoTimeMaybe();
    try {
      entries = FilesystemUtils.readdir(name);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, name);
    }
    Collection<Path> result = new ArrayList<>(entries.length);
    for (String entry : entries) {
      result.add(path.getChild(entry));
    }
    return result;
  }

  @Override
  protected PathFragment resolveOneLink(Path path) throws IOException {
    // Beware, this seemingly simple code belies the complex specification of
    // FileSystem.resolveOneLink().
    return stat(path, false).isSymbolicLink()
        ? readSymbolicLink(path)
        : null;
  }

  /**
   * Converts from {@link com.google.devtools.build.lib.unix.FilesystemUtils.Dirents.Type} to
   * {@link com.google.devtools.build.lib.vfs.Dirent.Type}.
   */
  private static Dirent.Type convertToDirentType(Dirents.Type type) {
    switch (type) {
      case FILE:
        return Dirent.Type.FILE;
      case DIRECTORY:
        return Dirent.Type.DIRECTORY;
      case SYMLINK:
        return Dirent.Type.SYMLINK;
      case UNKNOWN:
        return Dirent.Type.UNKNOWN;
      default:
        throw new IllegalArgumentException("Unknown type " + type);
    }
  }

  @Override
  protected Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      Dirents unixDirents = FilesystemUtils.readdir(name,
          followSymlinks ? ReadTypes.FOLLOW : ReadTypes.NOFOLLOW);
      Preconditions.checkState(unixDirents.hasTypes());
      List<Dirent> dirents = Lists.newArrayListWithCapacity(unixDirents.size());
      for (int i = 0; i < unixDirents.size(); i++) {
        dirents.add(new Dirent(unixDirents.getName(i),
            convertToDirentType(unixDirents.getType(i))));
      }
      return dirents;
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, name);
    }
  }

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    return statInternal(path, followSymlinks);
  }

  @VisibleForTesting
  protected UnixFileStatus statInternal(Path path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return new UnixFileStatus(followSymlinks
                                      ? FilesystemUtils.stat(name)
                                      : FilesystemUtils.lstat(name));
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  // Like stat(), but returns null instead of throwing.
  // This is a performance optimization in the case where clients
  // catch and don't re-throw.
  @Override
  protected FileStatus statNullable(Path path, boolean followSymlinks) {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      ErrnoFileStatus stat = followSymlinks
          ? FilesystemUtils.errnoStat(name)
          : FilesystemUtils.errnoLstat(name);
      return stat.hasError() ? null : new UnixFileStatus(stat);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    return statNullable(path, followSymlinks) != null;
  }

  /**
   * Return true iff the {@code stat} of {@code path} resulted in an {@code ENOENT}
   * or {@code ENOTDIR} error.
   */
  @Override
  protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      ErrnoFileStatus stat = followSymlinks
          ? FilesystemUtils.errnoStat(name)
          : FilesystemUtils.errnoLstat(name);
      if (!stat.hasError()) {
        return new UnixFileStatus(stat);
      }
      int errno = stat.getErrno();
      if (errno == ErrnoFileStatus.ENOENT || errno == ErrnoFileStatus.ENOTDIR) {
        return null;
      }
      // This should not return -- we are calling stat here just to throw the proper exception.
      // However, since there may be transient IO errors, we cannot guarantee that an exception will
      // be thrown.
      // TODO(bazel-team): Extract the exception-construction code and make it visible separately in
      // FilesystemUtils to avoid having to do a duplicate stat call.
      return stat(path, followSymlinks);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    return (statInternal(path, true).getPermissions() & 0400) != 0;
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    return (statInternal(path, true).getPermissions() & 0200) != 0;
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    return (statInternal(path, true).getPermissions() & 0100) != 0;
  }

  /**
   * Adds or remove the bits specified in "permissionBits" to the permission
   * mask of the file specified by {@code path}. If the argument {@code add} is
   * true, the specified permissions are added, otherwise they are removed.
   *
   * @throws IOException if there was an error writing the file's metadata
   */
  private void modifyPermissionBits(Path path, int permissionBits, boolean add)
    throws IOException {
    synchronized (path) {
      int oldMode = statInternal(path, true).getPermissions();
      int newMode = add ? (oldMode | permissionBits) : (oldMode & ~permissionBits);
      FilesystemUtils.chmod(path.toString(), newMode);
    }
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    modifyPermissionBits(path, 0400, readable);
  }

  @Override
  protected void setWritable(Path path, boolean writable) throws IOException {
    modifyPermissionBits(path, 0200, writable);
  }

  @Override
  protected void setExecutable(Path path, boolean executable) throws IOException {
    modifyPermissionBits(path, 0111, executable);
  }

  @Override
  protected void chmod(Path path, int mode) throws IOException {
    synchronized (path) {
      FilesystemUtils.chmod(path.toString(), mode);
    }
  }

  @Override
  public boolean supportsModifications() {
    return true;
  }

  @Override
  public boolean supportsSymbolicLinks() {
    return true;
  }

  @Override
  protected boolean createDirectory(Path path) throws IOException {
    synchronized (path) {
      // Note: UNIX mkdir(2), FilesystemUtils.mkdir() and createDirectory all
      // have different ways of representing failure!
      if (FilesystemUtils.mkdir(path.toString(), 0777)) {
        return true; // successfully created
      }

      // false => EEXIST: something is already in the way (file/dir/symlink)
      if (isDirectory(path, false)) {
        return false; // directory already existed
      } else {
        throw new IOException(path + " (File exists)");
      }
    }
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment)
      throws IOException {
    SymlinkImplementation strategy = computeSymlinkImplementation(linkPath, targetFragment);
    switch (strategy) {
      case HARDLINK:
        FilesystemUtils.link(targetFragment.toString(), linkPath.toString());
        break;

      case JUNCTION:  // Junctions are emulated on Linux with symlinks, fall through
      case SYMLINK:
        synchronized (linkPath) {
          FilesystemUtils.symlink(targetFragment.toString(), linkPath.toString());
        }
        break;

      case FAIL:
        if (symlinkLogFile == null) {
          // Otherwise, it was logged in computeSymlinkImplementation().
          throw new IOException(String.format("Symlink emulation failed for symlink: %s -> %s",
              linkPath, targetFragment));
        }
    }
  }

  private boolean isHardLinkAllowed(Path path) {
    for (Path root : rootsWithAllowedHardlinks) {
      if (path.startsWith(root)) {
        return true;
      }
    }

    return false;
  }

  private static final int JVM_ID = new Random().nextInt(10000);

  private void writeLogMessage(String message) {
    String logLine = String.format("[%04d] %s\n", JVM_ID, message);
    // FileLock does not work for synchronization between threads in the same JVM as per its Javadoc
    synchronized (symlinkLogFile) {

      try (FileChannel channel = new RandomAccessFile(symlinkLogFile, "rwd").getChannel()) {
        try (FileLock lock = channel.lock()) {
          channel.position(channel.size());
          ByteBuffer data = Charset.forName("UTF-8").newEncoder().encode(CharBuffer.wrap(logLine));
          channel.write(data);
        }
      } catch (IOException e) {
        // Not much intelligent we can do here
      }
    }
  }

  /**
   * How to create a particular symbolic link.
   *
   * <p>Necessary because Windows doesn't support symlinks properly, so we have to work around it.
   * No, even though they say <i>"Microsoft has implemented its symbolic links to function just like
   * UNIX links"</i>, it's a lie.
   */
  private enum SymlinkImplementation {
    /**
     * We can't emulate this link. Fail.
     */
    FAIL,

    /**
     * Create a hard link. This only works if we have write access to the ultimate destination on
     * the link.
     */
    HARDLINK,

    /**
     * Create a junction. This only works if the ultimate target of the "symlink" is a directory.
     */
    JUNCTION,

    /**
     * Use a symlink. Always works, but only on Unix-based operating systems.
     */
    SYMLINK,
  }

  private SymlinkImplementation emitSymlinkCompatibilityMessage(
    String reason, Path linkPath, PathFragment targetFragment) {
    if (symlinkLogFile == null) {
      return SymlinkImplementation.FAIL;
    }

    Exception e = new Exception();
    e.fillInStackTrace();
    String msg = String.format("ILLEGAL (%s): %s -> %s\nStack:\n%s",
        reason, linkPath.getPathString(), targetFragment.getPathString(),
        Throwables.getStackTraceAsString(e));
    writeLogMessage(msg);
    return SymlinkImplementation.SYMLINK;  // We are in logging mode, pretend everything is A-OK
  }

  private SymlinkImplementation computeSymlinkImplementation(
      Path linkPath, PathFragment targetFragment) throws IOException {
    if (symlinkStrategy != SymlinkStrategy.WINDOWS_COMPATIBLE) {
      return SymlinkImplementation.SYMLINK;
    }

    Path targetPath = linkPath.getRelative(targetFragment);
    if (!targetPath.exists(Symlinks.FOLLOW)) {
      return emitSymlinkCompatibilityMessage(
          "Target does not exist", linkPath, targetFragment);
    }

    targetPath = targetPath.resolveSymbolicLinks();
    if (targetPath.isDirectory(Symlinks.FOLLOW)) {
      // We can create junctions to any directory.
      return SymlinkImplementation.JUNCTION;
    }

    if (isHardLinkAllowed(targetPath)) {
      // We have write access to the destination and it's a file, so we can do this
      return SymlinkImplementation.HARDLINK;
    }

    return emitSymlinkCompatibilityMessage(
        "Target is a non-writable file", linkPath, targetFragment);
  }

  public SymlinkStrategy getSymlinkStrategy() {
    return symlinkStrategy;
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    // Note that the default implementation of readSymbolicLinkUnchecked calls this method and thus
    // is optimal since we only make one system call in here.
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return new PathFragment(FilesystemUtils.readlink(name));
    } catch (IOException e) {
      // EINVAL => not a symbolic link.  Anything else is a real error.
      throw e.getMessage().endsWith("(Invalid argument)") ? new NotASymlinkException(path) : e;
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_READLINK, name);
    }
  }

  @Override
  protected void renameTo(Path sourcePath, Path targetPath) throws IOException {
    synchronized (sourcePath) {
      FilesystemUtils.rename(sourcePath.toString(), targetPath.toString());
    }
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getSize();
  }

  @Override
  protected boolean delete(Path path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    synchronized (path) {
      try {
        return FilesystemUtils.remove(name);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, name);
      }
    }
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  protected void setLastModifiedTime(Path path, long newTime) throws IOException {
    synchronized (path) {
      if (newTime == -1L) { // "now"
        FilesystemUtils.utime(path.toString(), true, 0);
      } else {
        // newTime > MAX_INT => -ve unixTime
        int unixTime = (int) (newTime / 1000);
        FilesystemUtils.utime(path.toString(), false, unixTime);
      }
    }
  }

  @Override
  protected byte[] getxattr(Path path, String name) throws IOException {
    String pathName = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return FilesystemUtils.getxattr(pathName, name);
    } catch (UnsupportedOperationException e) {
      // getxattr() syscall is not supported by the underlying filesystem (it returned ENOTSUP).
      // Per method contract, treat this as ENODATA.
      return null;
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_XATTR, pathName);
    }
  }

  @Override
  protected byte[] getMD5Digest(Path path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return FilesystemUtils.md5sum(name).asBytes();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_MD5, name);
    }
  }
}
