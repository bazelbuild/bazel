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

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.jni.JniLoader;
import javax.annotation.Nullable;

/** Utility class providing access to POSIX filesystem calls via JNI. */
final class NativePosixFiles {

  static {
    JniLoader.loadJni();
  }

  private NativePosixFiles() {}

  /**
   * Native wrapper around Linux readlink(2) call.
   *
   * @param path the file of interest
   * @return the pathname to which the symbolic link 'path' links, or {@code null} if 'path' is not
   *     a symbolic link.
   * @throws NativePosixFilesException iff the readlink() call failed for any other reason
   */
  @Nullable
  static native String readlink(String path) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX chmod(2) syscall.
   *
   * @param path the file of interest
   * @param mode the POSIX type and permission mode bits to set
   * @throws NativePosixFilesException iff the chmod() call failed.
   */
  static native void chmod(String path, int mode) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX symlink(2) syscall.
   *
   * @param oldpath the file to link to
   * @param newpath the new path for the link
   * @throws NativePosixFilesException iff the symlink() syscall failed.
   */
  static native void symlink(String oldpath, String newpath) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX link(2) syscall.
   *
   * @param oldpath the file to link to
   * @param newpath the new path for the link
   * @throws NativePosixFilesException iff the link() syscall failed.
   */
  static native void link(String oldpath, String newpath) throws NativePosixFilesException;

  /** How stat() and lstat() should handle errors. */
  static final class StatErrorHandling {
    private final char code;

    private StatErrorHandling(char code) {
      this.code = code;
    }

    /** Always throw an exception. */
    static final StatErrorHandling ALWAYS_THROW = new StatErrorHandling('a');

    /** Throw an exception unless the error is ENOENT/ENOTDIR, in which case return null. */
    static final StatErrorHandling THROW_UNLESS_NOT_FOUND = new StatErrorHandling('f');

    /** Never throw an exception. Return null instead. */
    static final StatErrorHandling NEVER_THROW = new StatErrorHandling('n');

    private char getCode() {
      return code;
    }
  }

  /** File metadata, as returned by stat() or lstat(). */
  @SuppressWarnings("GoodTime")
  record Stat(int mode, long mtimeNanos, long ctimeNanos, long size, long ino) {}

  /**
   * Native wrapper around POSIX stat(2) syscall.
   *
   * @param path the file to stat.
   * @param errorHandling how to handle errors.
   * @return a {@link Stat} containing the metadata, or {@code null} if the syscall failed and
   *     {@code errorHandling} suppressed the exception.
   * @throws NativePosixFilesException if the stat() syscall failed.
   */
  @Nullable
  static Stat stat(String path, StatErrorHandling errorHandling) throws NativePosixFilesException {
    return stat(path, errorHandling.getCode());
  }

  @Nullable
  private static native Stat stat(String path, char errorHandling) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX lstat(2) syscall.
   *
   * @param path the file to lstat.
   * @param errorHandling how to handle errors.
   * @return a {@link Stat} containing the metadata, or {@code null} if the syscall failed and
   *     {@code errorHandling} suppressed the exception.
   * @throws NativePosixFilesException if the lstat() syscall failed.
   */
  @Nullable
  static Stat lstat(String path, StatErrorHandling errorHandling) throws NativePosixFilesException {
    return lstat(path, errorHandling.getCode());
  }

  @Nullable
  private static native Stat lstat(String path, char errorHandling)
      throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX utimensat(2) syscall.
   *
   * @param path the file whose modification time should be changed.
   * @param now if true, ignore {@code epochNanos} and use the current time.
   * @param epochNanos the file modification time in nanoseconds since the UNIX epoch.
   * @throws NativePosixFilesException if the operation failed.
   */
  @SuppressWarnings("GoodTime") // performance-sensitive
  static native void utimensat(String path, boolean now, long epochNanos)
      throws NativePosixFilesException;

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
   * @throws NativePosixFilesException if the mkdir() syscall failed for any other reason.
   */
  static native boolean mkdir(String path, int mode) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX opendir(2)/readdir(3)/closedir(3) syscalls.
   *
   * @param path the directory to read.
   * @return an array of {@link Dirent} objects, one for each directory entry, excluding {@code .}
   *     and {@code ..}.
   * @throws NativePosixFilesException if the opendir(), readdir() or closedir() calls failed for
   *     any reason.
   */
  static native Dirent[] readdir(String path) throws NativePosixFilesException;

  /** A directory entry and its corresponding type, as returned by readdir(). */
  record Dirent(String name, Type type) {
    /** The type of the directory entry. */
    static final class Type {
      /** Regular file. */
      static final Type FILE = new Type();

      /** Directory. */
      static final Type DIRECTORY = new Type();

      /** Symbolic link. */
      static final Type SYMLINK = new Type();

      /** Character special device. */
      static final Type CHAR = new Type();

      /** Block special device. */
      static final Type BLOCK = new Type();

      /** Named pipe. */
      static final Type FIFO = new Type();

      /** Unix domain socket. */
      static final Type SOCKET = new Type();

      /** Unknown type. */
      static final Type UNKNOWN = new Type();

      private Type() {}
    }
  }

  /**
   * Native wrapper around POSIX rename(2) syscall.
   *
   * @param oldpath the source location.
   * @param newpath the destination location.
   * @throws NativePosixFilesException if the rename failed for any reason.
   */
  static native void rename(String oldpath, String newpath) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX remove(3) C library call.
   *
   * @param path the file or directory to remove.
   * @return true iff the file was actually deleted by this call.
   * @throws NativePosixFilesException if the remove failed, but the file was present prior to the
   *     call.
   */
  static native boolean remove(String path) throws NativePosixFilesException;

  /**
   * Native wrapper around POSIX mkfifo(3) C library call.
   *
   * @param path the name of the pipe to create.
   * @param mode the mode with which to create the pipe.
   * @throws NativePosixFilesException if the mkfifo failed.
   */
  static native void mkfifo(String path, int mode) throws NativePosixFilesException;

  /**
   * Native wrapper around Linux getxattr(2) syscall.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if any, or null if extended
   *     attributes are unsupported by the filesystem or if no such extended attribute is defined
   * @throws NativePosixFilesException if the call failed for any other reason
   */
  @Nullable
  static native byte[] getxattr(String path, String name) throws NativePosixFilesException;

  /**
   * Native wrapper around Linux lgetxattr(2) syscall.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if any, or null if extended
   *     attributes are unsupported by the filesystem or if no such extended attribute is defined
   * @throws NativePosixFilesException if the call failed for any other reason
   */
  @Nullable
  static native byte[] lgetxattr(String path, String name) throws NativePosixFilesException;

  /**
   * Deletes all directory trees recursively beneath the given path, which is expected to be a
   * directory. Does not remove the top directory.
   *
   * @param dir the directory hierarchy to remove
   * @throws NativePosixFilesException if the hierarchy cannot be removed successfully or if the
   *     given path is not a directory
   */
  static native void deleteTreesBelow(String dir) throws NativePosixFilesException;

  /** Logs a path string that does not have a Latin-1 coder. Called from JNI. */
  private static void logBadPath(String path) {
    BugReport.sendNonFatalBugReport(
        new IllegalStateException("Path string does not have a Latin-1 coder: %s".formatted(path)));
  }
}
