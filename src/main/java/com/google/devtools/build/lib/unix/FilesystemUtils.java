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
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.UnixJniLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

/**
 * Utility methods for access to UNIX filesystem calls not exposed by the Java
 * SDK. Exception messages are selected to be consistent with those generated
 * by the java.io package where appropriate--see package javadoc for details.
 */
public final class FilesystemUtils {

  private FilesystemUtils() {}

  /**
   * Returns true iff the file identified by 'path' is a symbolic link. Has
   * similar semantics to POSIX stat(2) syscall, with all errors being mapped to
   * a false return.
   *
   * @param path the file of interest
   * @return true iff path exists, is accessible and is a symlink
   */
  public static boolean isSymbolicLink(File path) {
    try {
      return lstat(path.toString()).isSymbolicLink();
    } catch (IOException e) {
      return false;
    }
  }


  /**
   * Returns true iff the file identified by 'path' is a directory. Has
   * similar semantics to POSIX stat(2) syscall, with all errors being mapped to
   * a false return.
   *
   * @param path the file of interest
   * @return true iff path exists, is accessible and is a symlink
   */
  public static boolean isDirectory(String path) {
    try {
      return lstat(path).isDirectory();
    } catch (IOException e) {
      return false;
    }
  }


  /**
   * Marks the file or directory {@code path} as executable. (Non-atomic)
   *
   * @see File#setReadOnly
   *
   * @param path the file of interest
   * @throws FileAccessException if path can't be accessed
   * @throws FileNotFoundException if path doesn't exist
   * @throws IOException for other filesystem or path errors
   */
  public static void setExecutable(File path) throws IOException {
    String p = path.toString();
    chmod(p, stat(p).getPermissions() | FileStatus.S_IEXEC);
  }

  /**
   * Marks the file or directory {@code path} as owner writable. (Non-atomic)
   *
   * @see File#setReadOnly
   *
   * @param path the file of interest
   * @throws FileAccessException if path can't be accessed
   * @throws FileNotFoundException if path doesn't exist
   * @throws IOException for other filesystem or path errors
   */
  public static void setWritable(File path) throws IOException {
    String p = path.toString();
    chmod(p, stat(p).getPermissions() | FileStatus.S_IWUSR);
  }

  /**
   * Changes permissions of a file.
   *
   * @param path the file whose mode to change.
   * @param mode the mode bits within 07777, interpreted according to
   *   long-standing UNIX tradition.
   * @throws IOException if the chmod() syscall failed.
   */
  public static void chmod(File path, int mode) throws IOException {
    int mask = FileStatus.S_ISUID |
               FileStatus.S_ISGID |
               FileStatus.S_ISVTX |
               FileStatus.S_IRWXA;
    chmod(path.toString(), mode & mask);
  }

  /*
   * Native-based implementation
   */

  static {
    if (!java.nio.charset.Charset.defaultCharset().name().equals("ISO-8859-1")) {
      // Defer the Logger call, so we don't deadlock if this is called from Logger
      // initialization code.
      new Thread() {
        @Override
        public void run() {
          // wait (if necessary) until the logging system is initialized
          synchronized (LogManager.getLogManager()) {}
          Logger.getLogger("com.google.devtools.build.lib.unix.FilesystemUtils").log(Level.FINE,
              "WARNING: Default character set is not latin1; java.io.File and " +
              "com.google.devtools.build.lib.unix.FilesystemUtils will represent some filenames " +
              "differently.");
        }
      }.start();
    }
    UnixJniLoader.loadJni();
  }

  /**
   * Native wrapper around Linux readlink(2) call.
   *
   * @param path the file of interest
   * @return the pathname to which the symbolic link 'path' links
   * @throws IOException iff the readlink() call failed
   */
  public static native String readlink(String path) throws IOException;

  /**
   * Native wrapper around POSIX chmod(2) syscall: Changes the file access
   * permissions of 'path' to 'mode'.
   *
   * @param path the file of interest
   * @param mode the POSIX type and permission mode bits to set
   * @throws IOException iff the chmod() call failed.
   */
  public static native void chmod(String path, int mode) throws IOException;

  /**
   * Native wrapper around POSIX symlink(2) syscall.
   *
   * @param oldpath the file to link to
   * @param newpath the new path for the link
   * @throws IOException iff the symlink() syscall failed.
   */
  public static native void symlink(String oldpath, String newpath)
      throws IOException;

  /**
   * Native wrapper around POSIX link(2) syscall.
   *
   * @param oldpath the file to link to
   * @param newpath the new path for the link
   * @throws IOException iff the link() syscall failed.
   */
  public static native void link(String oldpath, String newpath) throws IOException;

  /**
   * Native wrapper around POSIX stat(2) syscall.
   *
   * @param path the file to stat.
   * @return a FileStatus instance containing the metadata.
   * @throws IOException if the stat() syscall failed.
   */
  public static native FileStatus stat(String path) throws IOException;

  /**
   * Native wrapper around POSIX lstat(2) syscall.
   *
   * @param path the file to lstat.
   * @return a FileStatus instance containing the metadata.
   * @throws IOException if the lstat() syscall failed.
   */
  public static native FileStatus lstat(String path) throws IOException;

  /**
   * Native wrapper around POSIX stat(2) syscall.
   *
   * @param path the file to stat.
   * @return an ErrnoFileStatus instance containing the metadata.
   *   If there was an error, the return value's hasError() method
   *   will return true, and all stat information is undefined.
   */
  public static native ErrnoFileStatus errnoStat(String path);

  /**
   * Native wrapper around POSIX lstat(2) syscall.
   *
   * @param path the file to lstat.
   * @return an ErrnoFileStatus instance containing the metadata.
   *   If there was an error, the return value's hasError() method
   *   will return true, and all stat information is undefined.
   */
  public static native ErrnoFileStatus errnoLstat(String path);

  /**
   * Native wrapper around POSIX utime(2) syscall.
   *
   * Note: negative file times are interpreted as unsigned time_t.
   *
   * @param path the file whose times to change.
   * @param now if true, ignore actime/modtime parameters and use current time.
   * @param modtime the file modification time in seconds since the UNIX epoch.
   * @throws IOException if the utime() syscall failed.
   */
  public static native void utime(String path, boolean now, int modtime) throws IOException;

  /**
   * Native wrapper around POSIX mkdir(2) syscall.
   *
   * Caveat: errno==EEXIST is mapped to the return value "false", not
   * IOException.  It requires an additional stat() to determine if mkdir
   * failed because the directory already exists.
   *
   * @param path the directory to create.
   * @param mode the mode with which to create the directory.
   * @return true if the directory was successfully created; false if the
   *   system call returned EEXIST because some kind of a file (not necessarily
   *   a directory) already exists.
   * @throws IOException if the mkdir() syscall failed for any other reason.
   */
  public static native boolean mkdir(String path, int mode)
      throws IOException;

  /**
   * Native wrapper around POSIX opendir(2)/readdir(3)/closedir(3) syscall.
   *
   * @param path the directory to read.
   * @return the list of directory entries in the order they were returned by
   *   the system, excluding "." and "..".
   * @throws IOException if the call to opendir failed for any reason.
   */
  public static String[] readdir(String path) throws IOException {
    return readdir(path, ReadTypes.NONE).names;
  }

  /**
   * An enum for specifying now the types of the individual entries returned by
   * {@link #readdir(String, ReadTypes)} is to be returned.
   */
  public enum ReadTypes {
    NONE('n'),      // Do not read types
    NOFOLLOW('d'),  // Do not follow symlinks
    FOLLOW('f');    // Follow symlinks; never returns "SYMLINK" and returns "UNKNOWN" when dangling

    private final char code;

    private ReadTypes(char code) {
      this.code = code;
    }

    private char getCode() {
      return code;
    }
  }

  /**
   * A compound return type for readdir(), analogous to struct dirent[] in C. A low memory profile
   * is critical for this class, as instances are expected to be kept around for caching for
   * potentially a long time.
   */
  public static final class Dirents {

  /**
   * The type of the directory entry.
   */
  public enum Type {
    FILE,
    DIRECTORY,
    SYMLINK,
    UNKNOWN;

    private static Type forChar(char c) {
      if (c == 'f') {
        return Type.FILE;
      } else if (c == 'd') {
        return Type.DIRECTORY;
      } else if (c == 's') {
        return Type.SYMLINK;
      } else {
        return Type.UNKNOWN;
      }
    }
  }

    /** The names of the entries in a directory. */
    private final String[] names;
    /**
     * An optional (nullable) array of entry types, corresponding positionally
     * to the "names" field.  The types are:
     *   'd': a subdirectory
     *   'f': a regular file
     *   's': a symlink (only returned with {@code NOFOLLOW})
     *   '?': anything else
     * Note that unlike libc, this implementation of readdir() follows
     * symlinks when determining these types.
     *
     * <p>This is intentionally a byte array rather than a array of enums to save memory.
     */
    private final byte[] types;

    /** called from JNI */
    public Dirents(String[] names, byte[] types) {
      this.names = names;
      this.types = types;
    }

    public int size() {
      return names.length;
    }

    public boolean hasTypes() {
      return types != null;
    }

    public String getName(int i) {
      return names[i];
    }

    public Type getType(int i) {
      return Type.forChar((char) types[i]);
    }
  }

  /**
   * Native wrapper around POSIX opendir(2)/readdir(3)/closedir(3) syscall.
   *
   * @param path the directory to read.
   * @param readTypes How the types of individual entries should be returned. If {@code NONE},
   *   the "types" field in the result will be null.
   * @return a Dirents object, containing "names", the list of directory entries
   *   (excluding "." and "..") in the order they were returned by the system,
   *   and "types", an array of entry types (file, directory, etc) corresponding
   *   positionally to "names".
   * @throws IOException if the call to opendir failed for any reason.
   */
  public static Dirents readdir(String path, ReadTypes readTypes) throws IOException {
    // Passing enums to native code is possible, but onerous; we use a char instead.
    return readdir(path, readTypes.getCode());
  }

  private static native Dirents readdir(String path, char typeCode)
      throws IOException;

  /**
   * Native wrapper around POSIX rename(2) syscall.
   *
   * @param oldpath the source location.
   * @param newpath the destination location.
   * @throws IOException if the rename failed for any reason.
   */
  public static native void rename(String oldpath, String newpath)
      throws IOException;

  /**
   * Native wrapper around POSIX remove(3) C library call.
   *
   * @param path the file or directory to remove.
   * @return true iff the file was actually deleted by this call.
   * @throws IOException if the remove failed, but the file was present prior to the call.
   */
  public static native boolean remove(String path) throws IOException;

  /**
   * Native wrapper around POSIX mkfifo(3) C library call.
   *
   * @param path the name of the pipe to create.
   * @param mode the mode with which to create the pipe.
   * @throws IOException if the mkfifo failed.
   */
  @VisibleForTesting
  public static native void mkfifo(String path, int mode) throws IOException;

  /********************************************************************
   *                                                                  *
   *                  Linux extended file attributes                  *
   *                                                                  *
   ********************************************************************/

  /**
   * Native wrapper around Linux getxattr(2) syscall.
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if
   *   any, or null if no such attribute is defined (ENODATA).
   * @throws IOException if the call failed for any other reason.
   */
  public static native byte[] getxattr(String path, String name)
      throws IOException;

  /**
   * Native wrapper around Linux lgetxattr(2) syscall.  (Like getxattr, but
   * does not follow symbolic links.)
   *
   * @param path the file whose extended attribute is to be returned.
   * @param name the name of the extended attribute key.
   * @return the value of the extended attribute associated with 'path', if
   *   any, or null if no such attribute is defined (ENODATA).
   * @throws IOException if the call failed for any other reason.
   */
  public static native byte[] lgetxattr(String path, String name)
      throws IOException;

  /**
   * Returns the MD5 digest of the specified file, following symbolic links.
   *
   * @param path the file whose MD5 digest is required.
   * @return the MD5 digest, as a 16-byte array.
   * @throws IOException if the call failed for any reason.
   */
  static native byte[] md5sumAsBytes(String path) throws IOException;

  /**
   * Returns the MD5 digest of the specified file, following symbolic links.
   *
   * @param path the file whose MD5 digest is required.
   * @return the MD5 digest, as a {@link HashCode}
   * @throws IOException if the call failed for any reason.
   */
  public static HashCode md5sum(String path) throws IOException {
    return HashCode.fromBytes(md5sumAsBytes(path));
  }

  /**
   * Removes entire directory tree. Doesn't follow symlinks.
   *
   * @param path the file or directory to remove.
   * @throws IOException if the remove failed.
   */
  public static void rmTree(String path) throws IOException {
    if (isDirectory(path)) {
      String[] contents = readdir(path);
      for (String entry : contents) {
        rmTree(path + "/" + entry);
      }
    }
    remove(path.toString());
  }
}
