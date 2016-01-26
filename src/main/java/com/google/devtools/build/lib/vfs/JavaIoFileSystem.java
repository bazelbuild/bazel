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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.unix.FileAccessException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collection;

/**
 * A FileSystem that does not use any JNI and hence, does not require a shared library be present at
 * execution.
 *
 * <p>Note: Blaze profiler tasks are defined on the system call level - thus we do not distinguish
 * (from profiling perspective) between different methods on this class that end up doing stat()
 * system call - they all are associated with the VFS_STAT task.
 */
@ThreadSafe
public class JavaIoFileSystem extends AbstractFileSystemWithCustomStat {
  private static final LinkOption[] NO_LINK_OPTION = new LinkOption[0];
  // This isn't generally safe; we rely on the file system APIs not modifying the array.
  private static final LinkOption[] NOFOLLOW_LINKS_OPTION =
      new LinkOption[] { LinkOption.NOFOLLOW_LINKS };

  protected static final String ERR_IS_DIRECTORY = " (Is a directory)";
  protected static final String ERR_DIRECTORY_NOT_EMPTY = " (Directory not empty)";
  protected static final String ERR_FILE_EXISTS = " (File exists)";
  protected static final String ERR_NO_SUCH_FILE_OR_DIR = " (No such file or directory)";
  protected static final String ERR_NOT_A_DIRECTORY = " (Not a directory)";

  protected File getIoFile(Path path) {
    return new File(path.toString());
  }

  private LinkOption[] linkOpts(boolean followSymlinks) {
    return followSymlinks ? NO_LINK_OPTION : NOFOLLOW_LINKS_OPTION;
  }

  @Override
  protected Collection<Path> getDirectoryEntries(Path path) throws IOException {
    File file = getIoFile(path);
    String[] entries = null;
    long startTime = Profiler.nanoTimeMaybe();
    try {
      entries = file.list();
      if (entries == null) {
        if (file.exists()) {
          throw new IOException(path + ERR_NOT_A_DIRECTORY);
        } else {
          throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
        }
      }
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_DIR, file.getPath());
    }
    Collection<Path> result = new ArrayList<>(entries.length);
    for (String entry : entries) {
      if (!entry.equals(".") && !entry.equals("..")) {
        result.add(path.getChild(entry));
      }
    }
    return result;
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return Files.exists(file.toPath(), linkOpts(followSymlinks));
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, path.toString());
    }
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    try {
      if (!file.exists()) {
        throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
      }
      return file.canRead();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, file.getPath());
    }
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    try {
      if (!file.exists()) {
        if (linkExists(file)) {
          throw new IOException(path + ERR_PERMISSION_DENIED);
        } else {
          throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
        }
      }
      return file.canWrite();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, file.getPath());
    }
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    try {
      if (!file.exists()) {
        throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
      }
      return file.canExecute();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, file.getPath());
    }
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    File file = getIoFile(path);
    if (!file.exists()) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }
    file.setReadable(readable);
  }

  @Override
  protected void setWritable(Path path, boolean writable) throws IOException {
    File file = getIoFile(path);
    if (!file.exists()) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }
    file.setWritable(writable);
  }

  @Override
  protected void setExecutable(Path path, boolean executable) throws IOException {
    File file = getIoFile(path);
    if (!file.exists()) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }
    file.setExecutable(executable);
  }

  @Override
  public boolean supportsModifications() {
    return true;
  }

  @Override
  public boolean supportsSymbolicLinksNatively() {
    return true;
  }

  @Override
  protected boolean createDirectory(Path path) throws IOException {

    // We always synchronize on the current path before doing it on the parent path and file system
    // path structure ensures that this locking order will never be reversed.
    // When refactoring, check that subclasses still work as expected and there can be no
    // deadlocks.
    synchronized (path) {
      File file = getIoFile(path);
      if (file.mkdir()) {
        return true;
      }

      // We will be checking the state of the parent path as well. Synchronize on it before
      // attempting anything.
      Path parentDirectory = path.getParentDirectory();
      synchronized (parentDirectory) {
        if (fileIsSymbolicLink(file)) {
          throw new IOException(path + ERR_FILE_EXISTS);
        }
        if (file.isDirectory()) {
          return false; // directory already existed
        } else if (file.exists()) {
          throw new IOException(path + ERR_FILE_EXISTS);
        } else if (!file.getParentFile().exists()) {
          throw new FileNotFoundException(path.getParentDirectory() + ERR_NO_SUCH_FILE_OR_DIR);
        }
        // Parent directory apparently exists - try to create our directory again - protecting
        // against the case where parent directory would be created right before us obtaining
        // synchronization lock.
        if (file.mkdir()) {
          return true; // Everything is fine finally.
        } else if (!file.getParentFile().canWrite()) {
          throw new FileAccessException(path + ERR_PERMISSION_DENIED);
        } else {
          // Parent exists, is writable, yet we can't create our directory.
          throw new FileNotFoundException(path.getParentDirectory() + ERR_NOT_A_DIRECTORY);
        }
      }
    }
  }

  private boolean linkExists(File file) {
    String shortName = file.getName();
    File parentFile = file.getParentFile();
    if (parentFile == null) {
      return false;
    }
    String[] filenames = parentFile.list();
    if (filenames == null) {
      return false;
    }
    for (String name : filenames) {
      if (name.equals(shortName)) {
        return true;
      }
    }
    return false;
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment)
      throws IOException {
    File file = getIoFile(linkPath);
    try {
      Files.createSymbolicLink(file.toPath(), new File(targetFragment.getPathString()).toPath());
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
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    try {
      String link = Files.readSymbolicLink(file.toPath()).toString();
      return new PathFragment(link);
    } catch (java.nio.file.NotLinkException e) {
      throw new NotASymlinkException(path);
    } catch (java.nio.file.NoSuchFileException e) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_READLINK, file.getPath());
    }
  }

  @Override
  protected void renameTo(Path sourcePath, Path targetPath) throws IOException {
    synchronized (sourcePath) {
      File sourceFile = getIoFile(sourcePath);
      File targetFile = getIoFile(targetPath);
      if (!sourceFile.renameTo(targetFile)) {
        if (!sourceFile.exists()) {
          throw new FileNotFoundException(sourcePath + ERR_NO_SUCH_FILE_OR_DIR);
        }
        if (targetFile.exists()) {
          if (targetFile.isDirectory() && targetFile.list().length > 0) {
            throw new IOException(targetPath + ERR_DIRECTORY_NOT_EMPTY);
          } else if (sourceFile.isDirectory() && targetFile.isFile()) {
            throw new IOException(sourcePath + " -> " + targetPath + ERR_NOT_A_DIRECTORY);
          } else if (sourceFile.isFile() && targetFile.isDirectory()) {
            throw new IOException(sourcePath + " -> " + targetPath + ERR_IS_DIRECTORY);
          } else {
            throw new IOException(sourcePath + " -> " + targetPath  + ERR_PERMISSION_DENIED);
          }
        } else {
          throw new FileAccessException(sourcePath + " -> " + targetPath + ERR_PERMISSION_DENIED);
        }
      }
    }
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return stat(path, followSymlinks).getSize();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, path);
    }
  }

  @Override
  protected boolean delete(Path path) throws IOException {
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    synchronized (path) {
      try {
        if (file.delete()) {
          return true;
        }
        if (file.exists()) {
          if (file.isDirectory() && file.list().length > 0) {
            throw new IOException(path + ERR_DIRECTORY_NOT_EMPTY);
          } else {
            throw new IOException(path + ERR_PERMISSION_DENIED);
          }
        }
        return false;
      } finally {
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_DELETE, file.getPath());
      }
    }
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    File file = getIoFile(path);
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return stat(path, followSymlinks).getLastModifiedTime();
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, file.getPath());
    }
  }

  private boolean fileIsSymbolicLink(File file) {
    return Files.isSymbolicLink(file.toPath());
  }

  @Override
  protected void setLastModifiedTime(Path path, long newTime) throws IOException {
    File file = getIoFile(path);
    if (!file.setLastModified(newTime)) {
      if (!file.exists()) {
        throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
      } else if (!file.getParentFile().canWrite()) {
        throw new FileAccessException(path.getParentDirectory() + ERR_PERMISSION_DENIED);
      } else {
        throw new FileAccessException(path + ERR_PERMISSION_DENIED);
      }
    }
  }

  @Override
  protected byte[] getMD5Digest(Path path) throws IOException {
    String name = path.toString();
    long startTime = Profiler.nanoTimeMaybe();
    try {
      return super.getMD5Digest(path);
    } finally {
      profiler.logSimpleTask(startTime, ProfilerTask.VFS_MD5, name);
    }
  }

  /**
   * Returns the status of a file. See {@link Path#stat(Symlinks)} for
   * specification.
   *
   * <p>The default implementation of this method is a "lazy" one, based on
   * other accessor methods such as {@link #isFile}, etc. Subclasses may provide
   * more efficient specializations. However, we still try to follow Unix-like
   * semantics of failing fast in case of non-existent files (or in case of
   * permission issues).
   */
  @Override
  protected FileStatus stat(final Path path, final boolean followSymlinks) throws IOException {
    File file = getIoFile(path);
    final BasicFileAttributes attributes;
    try {
      attributes = Files.readAttributes(
          file.toPath(), BasicFileAttributes.class, linkOpts(followSymlinks));
    } catch (java.nio.file.FileSystemException e) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }
    FileStatus status =  new FileStatus() {
      @Override
      public boolean isFile() {
        return attributes.isRegularFile() || isSpecialFile();
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
        return attributes.isSymbolicLink();
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
  protected FileStatus statIfFound(Path path, boolean followSymlinks) {
    try {
      return stat(path, followSymlinks);
    } catch (FileNotFoundException e) {
      // JavaIoFileSystem#stat (incorrectly) only throws FileNotFoundException (because it calls
      // #getLastModifiedTime, which can only throw a FileNotFoundException), so we always hit this
      // codepath. Thus, this method will incorrectly not throw an exception for some filesystem
      // errors.
      return null;
    } catch (IOException e) {
      // If this codepath is ever hit, then this method should be rewritten to properly distinguish
      // between not-found exceptions and others.
      throw new IllegalStateException(e);
    }
  }
}
