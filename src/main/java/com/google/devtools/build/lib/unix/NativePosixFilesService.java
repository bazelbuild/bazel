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

import com.google.devtools.build.lib.runtime.BlazeService;
import java.io.IOException;

/** A {@link BlazeService} providing access to POSIX filesystem calls. */
public interface NativePosixFilesService extends BlazeService {
  /**
   * Native wrapper around Linux readlink(2) call.
   *
   * @param path the file of interest
   * @return the pathname to which the symbolic link 'path' links
   * @throws IOException iff the readlink() call failed
   */
  String readlink(String path) throws IOException;

  /**
   * Native wrapper around POSIX chmod(2) syscall.
   *
   * @param path the file of interest
   * @param mode the POSIX type and permission mode bits to set
   * @throws IOException iff the chmod() call failed.
   */
  void chmod(String path, int mode) throws IOException;

  /**
   * Native wrapper around POSIX symlink(2) syscall.
   *
   * @param oldpath the file to link to
   * @param newpath the new path for the link
   * @throws IOException iff the symlink() syscall failed.
   */
  void symlink(String oldpath, String newpath) throws IOException;

  /**
   * Native wrapper around POSIX link(2) syscall.
   *
   * @param oldpath the file to link to
   * @param newpath the new path for the link
   * @throws IOException iff the link() syscall failed.
   */
  void link(String oldpath, String newpath) throws IOException;

  /** How stat() and lstat() should handle errors. */
  enum StatErrorHandling {
    /** Always throw an exception. */
    ALWAYS_THROW('a'),
    /** Throw an exception unless the error is ENOENT/ENOTDIR, in which case return null. */
    THROW_UNLESS_NOT_FOUND('f'),
    /* Never throw an exception. Return null instead. */
    NEVER_THROW('n');

    private final char code;

    private StatErrorHandling(char code) {
      this.code = code;
    }

    char getCode() {
      return code;
    }
  }

  /** File metadata, as returned by stat() or lstat(). */
  record Stat(int mode, long mtime, long ctime, long size, long ino) {}

  /**
   * Native wrapper around POSIX stat(2) syscall.
   *
   * @param path the file to stat.
   * @param errorHandling how to handle errors.
   * @return a {@link Stat} containing the metadata.
   * @throws IOException if the stat() syscall failed.
   */
  Stat stat(String path, StatErrorHandling errorHandling) throws IOException;

  /**
   * Native wrapper around POSIX lstat(2) syscall.
   *
   * @param path the file to lstat.
   * @param errorHandling how to handle errors.
   * @return a {@link Stat} containing the metadata.
   * @throws IOException if the lstat() syscall failed.
   */
  Stat lstat(String path, StatErrorHandling errorHandling) throws IOException;

  /**
   * Native wrapper around POSIX utimensat(2) syscall.
   *
   * <p>Note that, even though utimensat(2) supports up to nanosecond precision, this interface only
   * allows millisecond precision, which is what Bazel uses internally.
   *
   * @param path the file whose modification time should be changed.
   * @param now if true, ignore {@code epochMilli} and use the current time.
   * @param epochMilli the file modification time in milliseconds since the UNIX epoch.
   * @throws IOException if the operation failed.
   */
  void utimensat(String path, boolean now, long epochMilli) throws IOException;

  /**
   * Native wrapper around POSIX mkdir(2) syscall.
   *
   * <p>Caveat: errno==EEXIST is mapped to the return value "false", not IOException. It requires an
   * additional stat() to determine if mkdir failed because the directory already exists.
   *
   * @param path the directory to create.
   * @param mode the mode with which to create the directory.
   * @return true if the directory was successfully created; false if the system call returned
   *     EEXIST because some kind of a file (not necessarily a directory) already exists.
   * @throws IOException if the mkdir() syscall failed for any other reason.
   */
  boolean mkdir(String path, int mode) throws IOException;

  /**
   * Native wrapper around POSIX opendir(2)/readdir(3)/closedir(3) syscalls.
   *
   * @param path the directory to read.
   * @return an array of {@link Dirent} objects, one for each directory entry, excluding {@code .}
   *     and {@code ..}.
   * @throws IOException if the opendir(), readdir() or closedir() calls failed for any reason.
   */
  Dirent[] readdir(String path) throws IOException;

  /** A directory entry and its corresponding type, as returned by readdir(). */
  record Dirent(String name, Type type) {
    /** The type of the directory entry. */
    enum Type {
      /** Regular file. */
      FILE,
      /** Directory. */
      DIRECTORY,
      /** Symbolic link. */
      SYMLINK,
      /** Character special device. */
      CHAR,
      /* Block special device. */
      BLOCK,
      /** Named pipe. */
      FIFO,
      /** Unix domain socket. */
      SOCKET,
      /** Unknown type. */
      UNKNOWN
    }
  }

  /**
   * Native wrapper around POSIX rename(2) syscall.
   *
   * @param oldpath the source location.
   * @param newpath the destination location.
   * @throws IOException if the rename failed for any reason.
   */
  void rename(String oldpath, String newpath) throws IOException;

  /**
   * Native wrapper around POSIX remove(3) C library call.
   *
   * @param path the file or directory to remove.
   * @return true iff the file was actually deleted by this call.
   * @throws IOException if the remove failed, but the file was present prior to the call.
   */
  boolean remove(String path) throws IOException;

  /**
   * Native wrapper around POSIX mkfifo(3) C library call.
   *
   * @param path the name of the pipe to create.
   * @param mode the mode with which to create the pipe.
   * @throws IOException if the mkfifo failed.
   */
  void mkfifo(String path, int mode) throws IOException;

  /**
   * Native wrapper around Linux getxattr(2) syscall.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if any, or null if no such
   *     attribute is defined (ENODATA).
   * @throws IOException if the call failed for any other reason.
   */
  byte[] getxattr(String path, String name) throws IOException;

  /**
   * Native wrapper around Linux lgetxattr(2) syscall.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if any, or null if no such
   *     attribute is defined (ENODATA).
   * @throws IOException if the call failed for any other reason.
   */
  byte[] lgetxattr(String path, String name) throws IOException;

  /**
   * Deletes all directory trees recursively beneath the given path, which is expected to be a
   * directory. Does not remove the top directory.
   *
   * @param dir the directory hierarchy to remove
   * @throws IOException if the hierarchy cannot be removed successfully or if the given path is not
   *     a directory
   */
  void deleteTreesBelow(String dir) throws IOException;
}
