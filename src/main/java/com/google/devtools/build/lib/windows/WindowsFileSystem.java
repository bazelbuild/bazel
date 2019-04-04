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
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.windows.jni.WindowsFileOperations;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.DosFileAttributes;

/** File system implementation for Windows. */
@ThreadSafe
public class WindowsFileSystem extends JavaIoFileSystem {

  public static final LinkOption[] NO_OPTIONS = new LinkOption[0];
  public static final LinkOption[] NO_FOLLOW = new LinkOption[] {LinkOption.NOFOLLOW_LINKS};

  public WindowsFileSystem() throws DefaultHashFunctionNotSetException {}

  public WindowsFileSystem(DigestHashFunction hashFunction) {
    super(hashFunction);
  }

  @Override
  public String getFileSystemType(Path path) {
    // TODO(laszlocsomor): implement this properly, i.e. actually query this information from
    // somewhere (java.nio.Filesystem? System.getProperty? implement JNI method and use WinAPI?).
    return "ntfs";
  }

  @Override
  public boolean delete(Path path) throws IOException {
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return WindowsFileOperations.deletePath(path.getPathString());
    } catch (java.nio.file.DirectoryNotEmptyException e) {
      throw new IOException(path.getPathString() + ERR_DIRECTORY_NOT_EMPTY);
    } catch (java.nio.file.AccessDeniedException e) {
      throw new IOException(path.getPathString() + ERR_PERMISSION_DENIED);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, path.getPathString());
    }
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    Path targetPath =
        targetFragment.isAbsolute()
            ? getPath(targetFragment)
            : linkPath.getParentDirectory().getRelative(targetFragment);
    try {
      java.nio.file.Path link = getIoFile(linkPath).toPath();
      java.nio.file.Path target = getIoFile(targetPath).toPath();
      // Still Create a dangling junction if the target doesn't exist.
      if (!target.toFile().exists() || target.toFile().isDirectory()) {
        WindowsFileOperations.createJunction(link.toString(), target.toString());
      } else {
        Files.copy(target, link);
      }
    } catch (java.nio.file.FileAlreadyExistsException e) {
      throw new IOException(linkPath + ERR_FILE_EXISTS);
    } catch (java.nio.file.AccessDeniedException e) {
      throw new IOException(linkPath + ERR_PERMISSION_DENIED);
    } catch (java.nio.file.NoSuchFileException e) {
      throw new FileNotFoundException(linkPath + ERR_NO_SUCH_FILE_OR_DIR);
    }
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    java.nio.file.Path nioPath = getNioPath(path);
    return PathFragment.create(WindowsFileOperations.readSymlinkOrJunction(nioPath.toString()));
  }

  @Override
  public boolean supportsSymbolicLinksNatively(Path path) {
    return false;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return false;
  }

  @Override
  protected boolean fileIsSymbolicLink(File file) {
    try {
      if (isSymlinkOrJunction(file)) {
        return true;
      }
    } catch (IOException e) {
      // Did not work, try in another way
    }
    return super.fileIsSymbolicLink(file);
  }

  public static LinkOption[] symlinkOpts(boolean followSymlinks) {
    return followSymlinks ? NO_OPTIONS : NO_FOLLOW;
  }

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    File file = getIoFile(path);
    final DosFileAttributes attributes;
    try {
      attributes = getAttribs(file, followSymlinks);
    } catch (IOException e) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }

    final boolean isSymbolicLink = !followSymlinks && fileIsSymbolicLink(file);
    FileStatus status =
        new FileStatus() {
          @Override
          public boolean isFile() {
            return attributes.isRegularFile() || (isSpecialFile() && !isDirectory());
          }

          @Override
          public boolean isSpecialFile() {
            return attributes.isOther();
          }

          @Override
          public boolean isDirectory() {
            return attributes.isDirectory();
          }

          @Override
          public boolean isSymbolicLink() {
            return isSymbolicLink;
          }

          @Override
          public long getSize() throws IOException {
            return attributes.size();
          }

          @Override
          public long getLastModifiedTime() throws IOException {
            return attributes.lastModifiedTime().toMillis();
          }

          @Override
          public long getLastChangeTime() {
            // This is the best we can do with Java NIO...
            return attributes.lastModifiedTime().toMillis();
          }

          @Override
          public long getNodeId() {
            // TODO(bazel-team): Consider making use of attributes.fileKey().
            return -1;
          }
        };

    return status;
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    if (!followSymlinks) {
      try {
        if (isSymlinkOrJunction(getIoFile(path))) {
          return false;
        }
      } catch (IOException e) {
        return false;
      }
    }
    return super.isDirectory(path, followSymlinks);
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
  static boolean isSymlinkOrJunction(File file) throws IOException {
    return WindowsFileOperations.isSymlinkOrJunction(file.getPath());
  }

  private static DosFileAttributes getAttribs(File file, boolean followSymlinks)
      throws IOException {
    return Files.readAttributes(
        file.toPath(), DosFileAttributes.class, symlinkOpts(followSymlinks));
  }
}
