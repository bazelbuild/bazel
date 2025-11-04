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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.unix.NativePosixFiles.StatErrorHandling;
import com.google.devtools.build.lib.util.Blocker;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.AbstractFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.Collection;
import javax.annotation.Nullable;

/** This class implements the FileSystem interface using direct calls to the UNIX filesystem. */
@ThreadSafe
public class UnixFileSystem extends AbstractFileSystem {
  protected final String hashAttributeName;

  public UnixFileSystem(DigestHashFunction hashFunction, String hashAttributeName) {
    super(hashFunction);
    this.hashAttributeName = hashAttributeName;
  }

  @Override
  public Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    try {
      NativePosixFiles.Dirent[] dirents = NativePosixFiles.readdir(name);
      ImmutableList.Builder<String> builder = ImmutableList.builderWithExpectedSize(dirents.length);
      for (NativePosixFiles.Dirent dirent : dirents) {
        builder.add(dirent.name());
      }
      return builder.build();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, name);
    }
  }

  @Override
  @Nullable
  public PathFragment resolveOneLink(PathFragment path) throws IOException {
    // Beware, this seemingly simple code belies the complex specification of
    // FileSystem.resolveOneLink().
    return stat(path, false).isSymbolicLink() ? readSymbolicLink(path) : null;
  }

  /** Converts from {@link NativePosixFiles.Dirent.Type} to {@link Dirent.Type}. */
  private static Dirent.Type convertDirentType(NativePosixFiles.Dirent.Type type) {
    return switch (type) {
      case FILE -> Dirent.Type.FILE;
      case DIRECTORY -> Dirent.Type.DIRECTORY;
      case SYMLINK -> Dirent.Type.SYMLINK;
      default -> Dirent.Type.UNKNOWN;
    };
  }

  @Override
  public Collection<Dirent> readdir(PathFragment path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    try {
      NativePosixFiles.Dirent[] dirents = NativePosixFiles.readdir(name);
      ImmutableList.Builder<Dirent> builder = ImmutableList.builderWithExpectedSize(dirents.length);
      for (NativePosixFiles.Dirent dirent : dirents) {
        Dirent.Type type;
        // If the entry type is unknown, or if we're following symlinks and the entry is a symlink,
        // we need to stat the entry to get the type.
        if (dirent.type() == NativePosixFiles.Dirent.Type.UNKNOWN
            || (followSymlinks && dirent.type() == NativePosixFiles.Dirent.Type.SYMLINK)) {
          try {
            FileStatus stat = statIfFound(path.getRelative(dirent.name()), followSymlinks);
            type = stat != null ? ((UnixFileStatus) stat).getDirentType() : Dirent.Type.UNKNOWN;
          } catch (FileSymlinkLoopException e) {
            type = Dirent.Type.UNKNOWN;
          }
        } else {
          type = convertDirentType(dirent.type());
        }
        builder.add(new Dirent(dirent.name(), type));
      }
      return builder.build();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, name);
    }
  }

  @Override
  public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return followSymlinks
          ? NativePosixFiles.stat(name, StatErrorHandling.ALWAYS_THROW)
          : NativePosixFiles.lstat(name, StatErrorHandling.ALWAYS_THROW);
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
  public FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    String name = path.getPathString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return followSymlinks
          ? NativePosixFiles.stat(name, StatErrorHandling.NEVER_THROW)
          : NativePosixFiles.lstat(name, StatErrorHandling.NEVER_THROW);
    } catch (IOException e) {
      throw new IllegalStateException("unexpected exception", e);
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  @Override
  public boolean exists(PathFragment path, boolean followSymlinks) {
    return statNullable(path, followSymlinks) != null;
  }

  /**
   * Return true iff the {@code stat} of {@code path} resulted in an {@code ENOENT} or {@code
   * ENOTDIR} error.
   */
  @Override
  @Nullable
  public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    String name = path.getPathString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return followSymlinks
          ? NativePosixFiles.stat(name, StatErrorHandling.THROW_UNLESS_NOT_FOUND)
          : NativePosixFiles.lstat(name, StatErrorHandling.THROW_UNLESS_NOT_FOUND);
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, name);
    }
  }

  @Override
  public boolean isReadable(PathFragment path) throws IOException {
    return (stat(path, true).getPermissions() & 0400) != 0;
  }

  @Override
  public boolean isWritable(PathFragment path) throws IOException {
    return (stat(path, true).getPermissions() & 0200) != 0;
  }

  @Override
  public boolean isExecutable(PathFragment path) throws IOException {
    return (stat(path, true).getPermissions() & 0100) != 0;
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
    int oldMode = stat(path, true).getPermissions();
    int newMode = add ? (oldMode | permissionBits) : (oldMode & ~permissionBits);
    var comp = Blocker.begin();
    try {
      NativePosixFiles.chmod(path.toString(), newMode);
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  public void setReadable(PathFragment path, boolean readable) throws IOException {
    modifyPermissionBits(path, 0400, readable);
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    modifyPermissionBits(path, 0200, writable);
  }

  @Override
  public void setExecutable(PathFragment path, boolean executable) throws IOException {
    modifyPermissionBits(path, 0111, executable);
  }

  @Override
  public void chmod(PathFragment path, int mode) throws IOException {
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
  public boolean mayBeCaseOrNormalizationInsensitive() {
    return OS.getCurrent() == OS.DARWIN;
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
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    ArrayDeque<PathFragment> dirsToCreate = new ArrayDeque<>();
    for (PathFragment dir = path; dir != null; dir = dir.getParentDirectory()) {
      FileStatus stat = statIfFound(dir, /* followSymlinks= */ true);
      if (stat != null) {
        if (stat.isDirectory()) {
          break;
        } else {
          throw new IOException(path + " (File exists)");
        }
      }
      dirsToCreate.addLast(dir);
    }
    while (!dirsToCreate.isEmpty()) {
      var unused = createDirectory(dirsToCreate.removeLast());
    }
  }

  @Override
  public void createSymbolicLink(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
      throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.symlink(targetFragment.getSafePathString(), linkPath.toString());
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  public PathFragment readSymbolicLink(PathFragment path) throws IOException {
    // Note that the default implementation of readSymbolicLinkUnchecked calls this method and thus
    // is optimal since we only make one system call in here.
    String name = path.toString();
    long startTime = Profiler.instance().nanoTimeMaybe();
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
  public long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getSize();
  }

  @Override
  public boolean delete(PathFragment path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    var comp = Blocker.begin();
    try {
      return NativePosixFiles.remove(name);
    } finally {
      Blocker.end(comp);
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, name);
    }
  }

  @Override
  public long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return stat(path, followSymlinks).getLastModifiedTime();
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.utimensat(path.toString(), newTime == Path.NOW_SENTINEL_TIME, newTime);
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  @Nullable
  public byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    String pathName = path.toString();
    long startTime = Profiler.instance().nanoTimeMaybe();
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
  public byte[] getFastDigest(PathFragment path) throws IOException {
    // Attempt to obtain the digest from an extended attribute attached to the file. This is much
    // faster than reading and digesting the file's contents on the fly, especially for large files.
    return hashAttributeName.isEmpty() ? null : getxattr(path, hashAttributeName, true);
  }

  @Override
  public byte[] getDigest(PathFragment path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.instance().nanoTimeMaybe();
    try {
      return super.getDigest(path);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_MD5, name);
    }
  }

  @Override
  public void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    var comp = Blocker.begin();
    try {
      NativePosixFiles.link(originalPath.toString(), linkPath.toString());
    } finally {
      Blocker.end(comp);
    }
  }

  @Override
  public void deleteTreesBelow(PathFragment dir) throws IOException {
    if (isDirectory(dir, /* followSymlinks= */ false)) {
      long startTime = Profiler.instance().nanoTimeMaybe();
      var comp = Blocker.begin();
      try {
        NativePosixFiles.deleteTreesBelow(dir.toString());
      } finally {
        Blocker.end(comp);
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, dir.toString());
      }
    }
  }

  @Override
  public File getIoFile(PathFragment path) {
    return new File(StringEncoding.internalToPlatform(path.getPathString()));
  }

  @Override
  public java.nio.file.Path getNioPath(PathFragment path) {
    return java.nio.file.Path.of(StringEncoding.internalToPlatform(path.getPathString()));
  }

  @Override
  protected InputStream createFileInputStream(PathFragment path) throws IOException {
    return new FileInputStream(StringEncoding.internalToPlatform(path.getPathString()));
  }

  protected OutputStream createFileOutputStream(PathFragment path, boolean append)
      throws FileNotFoundException {
    return createFileOutputStream(path, append, /* internal= */ false);
  }

  @Override
  protected OutputStream createFileOutputStream(PathFragment path, boolean append, boolean internal)
      throws FileNotFoundException {
    String name = path.getPathString();
    if (!internal
        && profiler.isActive()
        && (profiler.isProfiling(ProfilerTask.VFS_WRITE)
            || profiler.isProfiling(ProfilerTask.VFS_OPEN))) {
      long startTime = Profiler.instance().nanoTimeMaybe();
      var comp = Blocker.begin();
      try {
        return new ProfiledFileOutputStream(name, /* append= */ append);
      } finally {
        Blocker.end(comp);
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_OPEN, name);
      }
    } else {
      var comp = Blocker.begin();
      try {
        return new FileOutputStream(StringEncoding.internalToPlatform(name), /* append= */ append);
      } finally {
        Blocker.end(comp);
      }
    }
  }

  private static final class ProfiledFileOutputStream extends FileOutputStream {
    private final String name;

    private ProfiledFileOutputStream(String name, boolean append) throws FileNotFoundException {
      super(StringEncoding.internalToPlatform(name), append);
      this.name = name;
    }

    @Override
    public void write(int b) throws IOException {
      long startTime = Profiler.instance().nanoTimeMaybe();
      try {
        super.write(b);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_WRITE, name);
      }
    }

    @Override
    public void write(byte[] b) throws IOException {
      long startTime = Profiler.instance().nanoTimeMaybe();
      try {
        super.write(b);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_WRITE, name);
      }
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      long startTime = Profiler.instance().nanoTimeMaybe();
      try {
        super.write(b, off, len);
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_WRITE, name);
      }
    }
  }
}
