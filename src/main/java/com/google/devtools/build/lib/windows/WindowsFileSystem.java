// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.windows;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileAccessException;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.DosFileAttributes;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** File system implementation for Windows. */
@ThreadSafe
public class WindowsFileSystem extends JavaIoFileSystem {

  public static final LinkOption[] NO_OPTIONS = new LinkOption[0];
  public static final LinkOption[] NO_FOLLOW = new LinkOption[] {LinkOption.NOFOLLOW_LINKS};

  private final boolean createSymbolicLinks;

  public WindowsFileSystem(DigestHashFunction hashFunction, boolean createSymbolicLinks) {
    super(hashFunction);
    this.createSymbolicLinks = createSymbolicLinks;
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    // TODO(laszlocsomor): implement this properly, i.e. actually query this information from
    // somewhere (java.nio.Filesystem? System.getProperty? implement JNI method and use WinAPI?).
    return "ntfs";
  }

  @Override
  public boolean delete(PathFragment path) throws IOException {
    Path nioPath;
    try {
      nioPath = getNioPath(path);
    } catch (InvalidPathException e) {
      return false;
    }

    long startTime = Profiler.instance().nanoTimeMaybe();
    try {
      return WindowsFileOperations.deletePath(nioPath.toString());
    } catch (java.nio.file.DirectoryNotEmptyException e) {
      throw new IOException(path.getPathString() + ERR_DIRECTORY_NOT_EMPTY, e);
    } catch (java.nio.file.AccessDeniedException e) {
      throw new IOException(path.getPathString() + ERR_PERMISSION_DENIED, e);
    } finally {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.VFS_DELETE, path.getPathString());
    }
  }

  @Override
  public void createSymbolicLink(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
      throws IOException {
    PathFragment targetPath =
        targetFragment.isAbsolute()
            ? targetFragment
            : (linkPath.getParentDirectory() != null
                ? linkPath.getParentDirectory().getRelative(targetFragment)
                : targetFragment);

    FileStatus stat = statIfFound(targetPath, /* followSymlinks= */ true);
    boolean existingFile = stat != null && stat.isFile();
    boolean existingDirectory = stat != null && stat.isDirectory();

    try {
      Path link = getNioPath(linkPath);
      Path target = getNioPath(targetPath);

      if (!createSymbolicLinks && existingFile) {
        // If symlinks aren't enabled and the target is an existing file, fall back to a copy.
        Files.copy(target, link);
      } else if (createSymbolicLinks
          && (existingFile || (!existingDirectory && type != SymlinkTargetType.DIRECTORY))) {
        // If symlinks are enabled and the target is not an existing or future directory, create a
        // symlink.
        WindowsFileOperations.createSymlink(link.toString(), target.toString());
      } else {
        // Otherwise, create a junction.
        WindowsFileOperations.createJunction(link.toString(), target.toString());
      }
    } catch (IOException e) {
      throw translateNioToIoException(linkPath, e);
    } catch (InvalidPathException e) {
      throw new IOException(linkPath.getPathString() + ERR_NO_SUCH_FILE_OR_DIR, e);
    }
  }

  @Override
  public PathFragment readSymbolicLink(PathFragment path) throws IOException {
    try {
      Path nioPath = getNioPath(path);
      return PathFragment.create(
          StringEncoding.platformToInternal(
              WindowsFileOperations.readSymlinkOrJunction(nioPath.toString())));
    } catch (InvalidPathException e) {
      FileNotFoundException fnfe = new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
      fnfe.initCause(e);
      throw fnfe;
    }
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return createSymbolicLinks;
  }

  @Override
  public boolean mayBeCaseOrNormalizationInsensitive() {
    return true;
  }

  @Override
  protected boolean fileIsSymbolicLink(Path file) {
    try {
      if (isSymlinkOrJunction(file)) {
        return true;
      }
    } catch (IOException | InvalidPathException e) {
      // Did not work, try in another way
    }
    return super.fileIsSymbolicLink(file);
  }

  public static LinkOption[] symlinkOpts(boolean followSymlinks) {
    return followSymlinks ? NO_OPTIONS : NO_FOLLOW;
  }

  @Override
  public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    final Path nioPath;
    final DosFileAttributes attributes;
    try {
      nioPath = getNioPath(path);
      attributes = getAttribs(nioPath, followSymlinks);
    } catch (AccessDeniedException e) {
      if (followSymlinks && mayBeSymlinkOrJunction(path)) {
        // Following a file symlink that (transitively) points to a directory fails with an access
        // denied error. Bazel creates such symlinks when the target doesn't exist yet at the time
        // the symlink is created. Keep treating them as dangling.
        FileNotFoundException fnfe = new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
        fnfe.initCause(e);
        throw fnfe;
      }
      // The path may well exist, but can't be accessed (yet or anymore), e.g. because of missing
      // permissions or because it is pending deletion while another process still holds a handle
      // to it. Reporting it as nonexistent would let callers such as Skyframe permanently record
      // incorrect information about the file system, so surface the actual error instead. This
      // matches the behavior of stat on Unix, which doesn't report EACCES as a missing file.
      FileAccessException fae = new FileAccessException(path + ERR_PERMISSION_DENIED);
      fae.initCause(e);
      throw fae;
    } catch (IOException | InvalidPathException e) {
      FileNotFoundException fnfe = new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
      fnfe.initCause(e);
      throw fnfe;
    }

    FileStatus status =
        new FileStatus() {
          @Nullable volatile Boolean isSymbolicLink; // null if not yet known
          volatile long lastChangeTimeNanos = -1;

          @Override
          public boolean isFile() {
            return !isSymbolicLink() && (attributes.isRegularFile() || isSpecialFile());
          }

          @Override
          public boolean isSpecialFile() {
            // attributes.isOther() returns false for symlinks but returns true for junctions.
            // Bazel treats junctions like symlinks. So let's return false here for junctions.
            // This fixes https://github.com/bazelbuild/bazel/issues/9176
            return !isSymbolicLink() && attributes.isOther();
          }

          @Override
          public boolean isDirectory() {
            return !isSymbolicLink() && attributes.isDirectory();
          }

          @Override
          public boolean isSymbolicLink() {
            if (isSymbolicLink == null) {
              isSymbolicLink = !followSymlinks && fileIsSymbolicLink(nioPath);
            }
            return isSymbolicLink;
          }

          @Override
          public long getSize() {
            return attributes.size();
          }

          @Override
          public long getLastModifiedTime() {
            return attributes.lastModifiedTime().toMillis();
          }

          @Override
          public long getLastModifiedTimeNanos() {
            return attributes.lastModifiedTime().to(TimeUnit.NANOSECONDS);
          }

          @Override
          public long getLastChangeTime() throws IOException {
            return Math.floorDiv(getLastChangeTimeNanos(), 1_000_000);
          }

          @Override
          public long getLastChangeTimeNanos() throws IOException {
            if (lastChangeTimeNanos == -1) {
              lastChangeTimeNanos =
                  WindowsFileOperations.getLastChangeTimeNanos(nioPath.toString(), followSymlinks);
            }
            return lastChangeTimeNanos;
          }

          @Override
          public long getNodeId() {
            // TODO(bazel-team): Consider making use of attributes.fileKey().
            return -1;
          }

          @Override
          public int getPermissions() {
            // Files on Windows are implicitly readable and executable.
            return 0555 | (attributes.isReadOnly() ? 0 : 0200);
          }
        };

    return status;
  }

  @Override
  public Collection<Dirent> readdir(PathFragment path, boolean followSymlinks) throws IOException {
    Collection<String> children = getDirectoryEntries(path);
    List<Dirent> dirents = new ArrayList<>(children.size());
    for (String child : children) {
      Dirent.Type type = Dirent.Type.UNKNOWN;
      try {
        FileStatus stat = statIfFound(path.getChild(child), followSymlinks);
        if (stat != null) {
          type = direntFromStat(stat);
        }
      } catch (IOException e) {
        // Intentionally ignored - a single entry that can't be stat'ed (e.g. because it is
        // inaccessible) shouldn't fail the entire listing. Report it as UNKNOWN instead.
      }
      dirents.add(new Dirent(child, type));
    }
    return dirents;
  }

  @Override
  public boolean isSymbolicLink(PathFragment path) throws IOException {
    try {
      return fileIsSymbolicLink(getNioPath(path));
    } catch (InvalidPathException e) {
      return false;
    }
  }

  @Override
  public boolean isDirectory(PathFragment path, boolean followSymlinks) throws IOException {
    if (!followSymlinks) {
      try {
        if (isSymlinkOrJunction(getNioPath(path))) {
          return false;
        }
      } catch (FileNotFoundException | InvalidPathException e) {
        return false;
      }
    }
    return super.isDirectory(path, followSymlinks);
  }

  @Override
  public void setReadable(PathFragment path, boolean readable) {
    // Windows does not have a notion of readable files.
    // https://github.com/openjdk/jdk/blob/e52a2aeeacaeb26c801b6e31f8e67e61b1ea2de3/src/java.base/windows/native/libjava/WinNTFileSystem_md.c#L473-L476
  }

  @Override
  public void setExecutable(PathFragment path, boolean executable) {
    // Windows does not have a notion of executable files.
    // https://github.com/openjdk/jdk/blob/e52a2aeeacaeb26c801b6e31f8e67e61b1ea2de3/src/java.base/windows/native/libjava/WinNTFileSystem_md.c#L473-L476
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    // Windows does not have a notion of read-only directories.
    // See https://learn.microsoft.com/en-us/windows/win32/fileio/file-attribute-constants.
    // JavaIoFileSystem#setWritable(dir, true) would throw, so reimplement it here as a no-op.
    if (isDirectory(path, /* followSymlinks= */ true)) {
      return;
    }
    super.setWritable(path, writable);
  }

  /**
   * Returns true if the path refers to a directory junction, directory symlink, or regular symlink.
   *
   * <p>Directory junctions are symbolic links created with "mklink /J" where the target is a
   * directory or another directory junction. Directory junctions can be created without any user
   * privileges.
   *
   * <p>Directory symlinks are symbolic links created with "mklink /D" where the target is a
   * directory or another directory symlink. Note that directory symlinks can only be created by
   * Administrators.
   *
   * <p>Normal symlinks are symbolic links created with "mklink". Normal symlinks should not point
   * at directories, because even though "mklink" can create the link, it will not be a functional
   * one (the linked directory's contents cannot be listed). Only Administrators may create regular
   * symlinks.
   *
   * <p>This method returns true for all three types as long as their target is a directory (even if
   * they are dangling), though only directory junctions and directory symlinks are useful.
   */
  @VisibleForTesting
  static boolean isSymlinkOrJunction(Path file) throws IOException {
    return WindowsFileOperations.isSymlinkOrJunction(file.toString());
  }

  /**
   * Returns whether the path is a symlink or junction, erring on the side of {@code true} if this
   * can't be determined.
   */
  private boolean mayBeSymlinkOrJunction(PathFragment path) {
    try {
      return isSymlinkOrJunction(getNioPath(path));
    } catch (IOException | InvalidPathException e) {
      return true;
    }
  }

  private static DosFileAttributes getAttribs(Path file, boolean followSymlinks)
      throws IOException {
    return Files.readAttributes(file, DosFileAttributes.class, symlinkOpts(followSymlinks));
  }
}
