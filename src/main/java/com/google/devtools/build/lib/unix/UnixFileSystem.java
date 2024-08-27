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
package com.google.devtools.build.lib.unix;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.unix.NativePosixFiles.Dirents;
import com.google.devtools.build.lib.unix.NativePosixFiles.ReadTypes;
import com.google.devtools.build.lib.util.Blocker;
import com.google.devtools.build.lib.vfs.AbstractFileSystemWithCustomStat;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/** This class implements the FileSystem interface using direct calls to the UNIX filesystem. */
@ThreadSafe
public class UnixFileSystem extends AbstractFileSystemWithCustomStat {
  protected final String hashAttributeName;

  public UnixFileSystem(DigestHashFunction hashFunction, String hashAttributeName) {
    super(hashFunction);
    this.hashAttributeName = hashAttributeName;
  }

  public static Dirent.Type getDirentFromMode(int mode) {
    if (com.google.devtools.build.lib.unix.FileStatus.isSpecialFile(mode)) {
      return Dirent.Type.UNKNOWN;
    } else if (com.google.devtools.build.lib.unix.FileStatus.isFile(mode)) {
      return Dirent.Type.FILE;
    } else if (com.google.devtools.build.lib.unix.FileStatus.isDirectory(mode)) {
      return Dirent.Type.DIRECTORY;
    } else if (com.google.devtools.build.lib.unix.FileStatus.isSymbolicLink(mode)) {
      return Dirent.Type.SYMLINK;
    } else {
      return Dirent.Type.UNKNOWN;
    }
  }

  /**
   * Eager implementation of FileStatus for file systems that have an atomic stat(2) syscall. A
   * proxy for {@link com.google.devtools.build.lib.unix.FileStatus}. Note that isFile and
   * getLastModifiedTime have slightly different meanings between UNIX and VFS.
   */
  @VisibleForTesting
  protected static class UnixFileStatus implements FileStatus {

    private final com.google.devtools.build.lib.unix.FileStatus status;

    UnixFileStatus(com.google.devtools.build.lib.unix.FileStatus status) {
      this.status = status;
    }

    @Override
    public boolean isFile() {
      return !isDirectory() && !isSymbolicLink();
    }

    @Override
    public boolean isDirectory() {
      return status.isDirectory();
    }

    @Override
    public boolean isSymbolicLink() {
      return status.isSymbolicLink();
    }

    @Override
    public boolean isSpecialFile() {
      return isFile() && !status.isRegularFile();
    }

    @Override
    public long getSize() {
      return status.getSize();
    }

    @Override
    public long getLastModifiedTime() {
      return (status.getLastModifiedTime() * 1000)
          + (status.getFractionalLastModifiedTime() / 1000000);
    }

    @Override
    public long getLastChangeTime() {
      return (status.getLastChangeTime() * 1000) + (status.getFractionalLastChangeTime() / 1000000);
    }

    @Override
    public long getNodeId() {
      // Note that we may want to include more information in this id number going forward,
      // especially the device number.
      return status.getInodeNumber();
    }

    @Override
    public int getPermissions() {
      return status.getPermissions();
    }

    @Override
    public String toString() {
      return status.toString();
    }
  }

  @Override
  protected Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    String name = path.getPathString();
    String[] entries;
    long startTime = Profiler.nanoTimeMaybe();
    try {
      entries = NativePosixFiles.readdir(name);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, name);
    }
    return Arrays.asList(entries);
  }

  @Override
  @Nullable
  protected PathFragment resolveOneLink(PathFragment path) throws IOException {
    // Beware, this seemingly simple code belies the complex specification of
    // FileSystem.resolveOneLink().
    return stat(path, false).isSymbolicLink() ? readSymbolicLink(path) : null;
  }

  /**
   * Converts from {@link com.google.devtools.build.lib.unix.NativePosixFiles.Dirents.Type} to
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
  protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
      throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      Dirents unixDirents =
          NativePosixFiles.readdir(name, followSymlinks ? ReadTypes.FOLLOW : ReadTypes.NOFOLLOW);
      Preconditions.checkState(unixDirents.hasTypes());
      List<Dirent> dirents = Lists.newArrayListWithCapacity(unixDirents.size());
      for (int i = 0; i < unixDirents.size(); i++) {
        dirents.add(
            new Dirent(unixDirents.getName(i), convertToDirentType(unixDirents.getType(i))));
      }
      return dirents;
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, name);
    }
  }

  @Override
  protected FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return statInternal(path, followSymlinks);
  }

  @VisibleForTesting
  protected UnixFileStatus statInternal(PathFragment path, boolean followSymlinks)
      throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return new UnixFileStatus(
          followSymlinks ? NativePosixFiles.stat(name) : NativePosixFiles.lstat(name));
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  // Like stat(), but returns null instead of throwing.
  // This is a performance optimization in the case where clients
  // catch and don't re-throw.
  @Override
  @Nullable
  protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      ErrnoFileStatus stat =
          followSymlinks ? NativePosixFiles.errnoStat(name) : NativePosixFiles.errnoLstat(name);
      return stat.hasError() ? null : new UnixFileStatus(stat);
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  @Override
  protected boolean exists(PathFragment path, boolean followSymlinks) {
    return statNullable(path, followSymlinks) != null;
  }

  /**
   * Return true iff the {@code stat} of {@code path} resulted in an {@code ENOENT} or {@code
   * ENOTDIR} error.
   */
  @Override
  @Nullable
  protected FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      ErrnoFileStatus stat =
          followSymlinks ? NativePosixFiles.errnoStat(name) : NativePosixFiles.errnoLstat(name);
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
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  @Override
  protected boolean isReadable(PathFragment path) throws IOException {
    return (statInternal(path, true).getPermissions() & 0400) != 0;
  }

  @Override
  protected boolean isWritable(PathFragment path) throws IOException {
    return (statInternal(path, true).getPermissions() & 0200) != 0;
  }

  @Override
  protected boolean isExecutable(PathFragment path) throws IOException {
    return (statInternal(path, true).getPermissions() & 0100) != 0;
  }

  /**
   * Adds or remove the bits specified in "permissionBits" to the permission mask of the file
   * specified by {@code path}. If the argument {@code add} is true, the specified permissions are
   * added, otherwise they are removed.
   *
   * @throws IOException if there was an error writing the file's metadata
   */
  private void modifyPermissionBits(PathFragment path, int permissionBits, boolean add)
      throws IOException {
    int oldMode = statInternal(path, true).getPermissions();
    int newMode = add ? (oldMode | permissionBits) : (oldMode & ~permissionBits);
    var comp = Blocker.begin();
    try {
      NativePosixFiles.chmod(path.toString(), newMode);
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  protected void setReadable(PathFragment path, boolean readable) throws IOException {
    modifyPermissionBits(path, 0400, readable);
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    modifyPermissionBits(path, 0200, writable);
  }

  @Override
  protected void setExecutable(PathFragment path, boolean executable) throws IOException {
    modifyPermissionBits(path, 0111, executable);
  }

  @Override
  protected void chmod(PathFragment path, int mode) throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.chmod(path.toString(), mode);
    } finally {
      Blocker.end(comp);
    }
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
    return true;
  }

  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    var comp = Blocker.begin();
    try {
      // Use 0777 so that the permissions can be overridden by umask(2).
      // Note: UNIX mkdir(2), FilesystemUtils.mkdir() and createDirectory all
      // have different ways of representing failure!
      if (NativePosixFiles.mkdir(path.toString(), 0777)) {
        return true; // successfully created
      }
    } finally {
      Blocker.end(comp);
    }

    // false => EEXIST: something is already in the way (file/dir/symlink)
    if (isDirectory(path, false)) {
      return false; // directory already existed
    } else {
      throw new IOException(path + " (File exists)");
    }
  }

  @Override
  protected boolean createWritableDirectory(PathFragment path) throws IOException {
    var comp = Blocker.begin();
    try {
      return NativePosixFiles.mkdirWritable(path.toString());
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    var comp = Blocker.begin();
    try {
      // Use 0777 so that the permissions can be overridden by umask(2).
      NativePosixFiles.mkdirs(path.toString(), 0777);
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  protected void createSymbolicLink(PathFragment linkPath, PathFragment targetFragment)
      throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.symlink(targetFragment.getSafePathString(), linkPath.toString());
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  protected PathFragment readSymbolicLink(PathFragment path) throws IOException {
    // Note that the default implementation of readSymbolicLinkUnchecked calls this method and thus
    // is optimal since we only make one system call in here.
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return PathFragment.create(NativePosixFiles.readlink(name));
    } catch (InvalidArgumentIOException e) {
      throw new NotASymlinkException(path, e);
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_READLINK, name);
    }
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.rename(sourcePath.toString(), targetPath.toString());
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  protected long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getSize();
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return NativePosixFiles.remove(name);
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, name);
    }
  }

  @Override
  protected long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    var comp = Blocker.begin();
    try {
      if (newTime == Path.NOW_SENTINEL_TIME) {
        NativePosixFiles.utime(path.toString(), true, 0);
      } else {
        // newTime > MAX_INT => -ve unixTime
        int unixTime = (int) (newTime / 1000);
        NativePosixFiles.utime(path.toString(), false, unixTime);
      }
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  @Nullable
  public byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    String pathName = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return followSymlinks
          ? NativePosixFiles.getxattr(pathName, name)
          : NativePosixFiles.lgetxattr(pathName, name);
    } catch (UnsupportedOperationException e) {
      // getxattr() syscall is not supported by the underlying filesystem (it returned ENOTSUP).
      // Per method contract, treat this as ENODATA.
      return null;
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_XATTR, pathName);
    }
  }

  @Override
  @Nullable
  protected byte[] getFastDigest(PathFragment path) throws IOException {
    // Attempt to obtain the digest from an extended attribute attached to the file. This is much
    // faster than reading and digesting the file's contents on the fly, especially for large files.
    return hashAttributeName.isEmpty() ? null : getxattr(path, hashAttributeName, true);
  }

  @Override
  protected byte[] getDigest(PathFragment path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return super.getDigest(path);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_MD5, name);
    }
  }

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.link(originalPath.toString(), linkPath.toString());
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  protected void deleteTreesBelow(PathFragment dir) throws IOException {
    if (isDirectory(dir, /*followSymlinks=*/ false)) {
      long startTime = Profiler.nanoTimeMaybe();
      var comp = Blocker.begin();
      try {
        NativePosixFiles.deleteTreesBelow(dir.toString());
      } finally {
        Blocker.end(comp);
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, dir.toString());
      }
    }
  }
}
