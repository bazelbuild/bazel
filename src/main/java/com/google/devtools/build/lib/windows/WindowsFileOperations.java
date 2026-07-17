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

import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.vfs.FileSystem.NotASymlinkException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.AccessDeniedException;

/** File operations on Windows. */
public class WindowsFileOperations {

  // A note about UNC paths and path prefixes on Windows. The prefixes can be:
  // - "\\?\", meaning it's a UNC path that is passed to user mode unicode WinAPI functions
  //   (e.g. CreateFileW) or a return value of theirs (e.g. GetLongPathNameW); this is the
  //   prefix we'll most often see
  // - "\??\", meaning it's Device Object path; it's mostly only used by kernel/driver functions
  //   but we may come across it when resolving junction targets, as the target's path is
  //   specified with this prefix, see usages of DeviceIoControl with FSCTL_GET_REPARSE_POINT
  // - "\\.\", meaning it's a Device Object path again; both "\??\" and "\\.\" are shorthands
  //   for the "\DosDevices\" Object Directory, so "\\.\C:" and "\??\C:" and "\DosDevices\C:"
  //   and "C:\" all mean the same thing, but functions like CreateFileW don't understand the
  //   fully qualified device path, only the shorthand versions; the difference between "\\.\"
  //   is "\??\" is not entirely clear (one is not available while Windows is booting, but
  //   that only concerns device drivers) but we most likely won't come across them anyway
  // Some of this is documented here:
  // - https://msdn.microsoft.com/en-us/library/windows/hardware/ff557762(v=vs.85).aspx
  // - https://msdn.microsoft.com/en-us/library/windows/hardware/ff565384(v=vs.85).aspx
  // - http://stackoverflow.com/questions/23041983
  // - http://stackoverflow.com/questions/14482421

  static {
    JniLoader.loadJni();
  }

  private WindowsFileOperations() {
    // Prevent construction
  }

  // Keep IS_SYMLINK_OR_JUNCTION_* values in sync with src/main/native/windows/file.cc.
  private static final int IS_SYMLINK_OR_JUNCTION_SUCCESS = 0;
  // IS_SYMLINK_OR_JUNCTION_ERROR = 1;
  private static final int IS_SYMLINK_OR_JUNCTION_DOES_NOT_EXIST = 2;

  // Keep GET_CHANGE_TIME_* values in sync with src/main/native/windows/file.cc.
  private static final int GET_CHANGE_TIME_SUCCESS = 0;
  //  private static final int GET_CHANGE_TIME_ERROR = 1;
  private static final int GET_CHANGE_TIME_DOES_NOT_EXIST = 2;
  private static final int GET_CHANGE_TIME_ACCESS_DENIED = 3;

  // Keep CREATE_JUNCTION_* values in sync with src/main/native/windows/file.h.
  private static final int CREATE_JUNCTION_SUCCESS = 0;
  // CREATE_JUNCTION_ERROR = 1;
  private static final int CREATE_JUNCTION_TARGET_NAME_TOO_LONG = 2;
  private static final int CREATE_JUNCTION_ALREADY_EXISTS_WITH_DIFFERENT_TARGET = 3;
  private static final int CREATE_JUNCTION_ALREADY_EXISTS_BUT_NOT_A_JUNCTION = 4;
  private static final int CREATE_JUNCTION_ACCESS_DENIED = 5;
  private static final int CREATE_JUNCTION_DISAPPEARED = 6;
  private static final int CREATE_JUNCTION_NOT_SUPPORTED = 7;

  // Keep CREATE_SYMLINK_* values in sync with src/main/native/windows/file.h.
  private static final int CREATE_SYMLINK_SUCCESS = 0;
  // CREATE_SYMLINK_ERROR = 1;
  private static final int CREATE_SYMLINK_TARGET_IS_DIRECTORY = 2;

  // Keep DELETE_PATH_* values in sync with src/main/native/windows/file.h.
  private static final int DELETE_PATH_SUCCESS = 0;
  // DELETE_PATH_ERROR = 1;
  private static final int DELETE_PATH_DOES_NOT_EXIST = 2;
  private static final int DELETE_PATH_DIRECTORY_NOT_EMPTY = 3;
  private static final int DELETE_PATH_ACCESS_DENIED = 4;

  // Keep READ_SYMLINK_OR_JUNCTION_* values in sync with src/main/native/windows/file.h.
  private static final int READ_SYMLINK_OR_JUNCTION_SUCCESS = 0;
  // READ_SYMLINK_OR_JUNCTION_ERROR = 1;
  private static final int READ_SYMLINK_OR_JUNCTION_ACCESS_DENIED = 2;
  private static final int READ_SYMLINK_OR_JUNCTION_DOES_NOT_EXIST = 3;
  private static final int READ_SYMLINK_OR_JUNCTION_NOT_A_LINK = 4;

  private static native int nativeIsSymlinkOrJunction(
      String path, boolean[] result, String[] error);

  private static native int nativeGetChangeTime(
      String path, boolean followReparsePoints, long[] result, String[] error);

  private static native int nativeCreateJunction(String name, String target, String[] error);

  private static native int nativeCreateSymlink(String name, String target, String[] error);

  private static native int nativeReadSymlinkOrJunction(
      String name, String[] result, String[] error);

  private static native int nativeDeletePath(String path, String[] error);

  /** Determines whether `path` is a junction point or directory symlink. */
  public static boolean isSymlinkOrJunction(String path) throws IOException {
    boolean[] result = new boolean[] {false};
    String[] error = new String[] {null};
    switch (nativeIsSymlinkOrJunction(WindowsPathOperations.asLongPath(path), result, error)) {
      case IS_SYMLINK_OR_JUNCTION_SUCCESS:
        return result[0];
      case IS_SYMLINK_OR_JUNCTION_DOES_NOT_EXIST:
        throw new FileNotFoundException(path);
      default:
        // This is IS_SYMLINK_OR_JUNCTION_ERROR (1). The JNI code puts a custom message in
        // 'error[0]'.
        break;
    }
    throw new IOException(String.format("Cannot tell if '%s' is link: %s", path, error[0]));
  }

  /** Returns the time at which the file was last changed, including metadata changes. */
  public static long getLastChangeTime(String path, boolean followReparsePoints)
      throws IOException {
    long[] result = new long[] {0};
    String[] error = new String[] {null};
    switch (nativeGetChangeTime(
        WindowsPathOperations.asLongPath(path), followReparsePoints, result, error)) {
      case GET_CHANGE_TIME_SUCCESS:
        return result[0];
      case GET_CHANGE_TIME_DOES_NOT_EXIST:
        throw new FileNotFoundException(path);
      case GET_CHANGE_TIME_ACCESS_DENIED:
        throw new AccessDeniedException(path);
      default:
        // This is GET_CHANGE_TIME_ERROR (1). The JNI code puts a custom message in 'error[0]'.
        break;
    }
    throw new IOException(String.format("Cannot get last change time of '%s': %s", path, error[0]));
  }

  /**
   * Creates a junction at `name`, pointing to `target`.
   *
   * <p>Both `name` and `target` may be Unix-style Windows paths (i.e. use forward slashes), and
   * they don't need to have a UNC prefix, not even if they are longer than `MAX_PATH`. The
   * underlying logic will take care of adding the prefixes if necessary.
   *
   * @throws IOException if some error occurs
   */
  public static void createJunction(String name, String target) throws IOException {
    String[] error = new String[] {null};
    switch (nativeCreateJunction(
        WindowsPathOperations.asLongPath(name), WindowsPathOperations.asLongPath(target), error)) {
      case CREATE_JUNCTION_SUCCESS:
        return;
      case CREATE_JUNCTION_TARGET_NAME_TOO_LONG:
        error[0] = "target name is too long";
        break;
      case CREATE_JUNCTION_ALREADY_EXISTS_WITH_DIFFERENT_TARGET:
        error[0] = "junction already exists with different target";
        break;
      case CREATE_JUNCTION_ALREADY_EXISTS_BUT_NOT_A_JUNCTION:
        error[0] = "a file or directory already exists at the junction's path";
        break;
      case CREATE_JUNCTION_ACCESS_DENIED:
        error[0] = "access is denied";
        break;
      case CREATE_JUNCTION_DISAPPEARED:
        error[0] = "the junction's path got modified unexpectedly";
        break;
      case CREATE_JUNCTION_NOT_SUPPORTED:
        error[0] = "filesystem does not support junctions";
        break;
      default:
        // This is CREATE_JUNCTION_ERROR (1). The JNI code puts a custom message in 'error[0]'.
        break;
    }
    throw new IOException(
        String.format("Cannot create junction (name=%s, target=%s): %s", name, target, error[0]));
  }

  public static void createSymlink(String name, String target) throws IOException {
    String[] error = new String[] {null};
    switch (nativeCreateSymlink(
        WindowsPathOperations.asLongPath(name), WindowsPathOperations.asLongPath(target), error)) {
      case CREATE_SYMLINK_SUCCESS:
        return;
      case CREATE_SYMLINK_TARGET_IS_DIRECTORY:
        error[0] = "symlink target is a directory, use a junction";
        break;
      default:
        // this is CREATE_SYMLINK_ERROR (1). The JNI code puts a custom message in 'error[0]'.
        break;
    }
    throw new IOException(
        String.format("Cannot create symlink (name=%s, target=%s): %s", name, target, error[0]));
  }

  public static String readSymlinkOrJunction(String name) throws IOException {
    String[] target = new String[] {null};
    String[] error = new String[] {null};
    switch (nativeReadSymlinkOrJunction(WindowsPathOperations.asLongPath(name), target, error)) {
      case READ_SYMLINK_OR_JUNCTION_SUCCESS:
        return WindowsPathOperations.removeUncPrefixAndUseSlashes(target[0]);
      case READ_SYMLINK_OR_JUNCTION_ACCESS_DENIED:
        throw new AccessDeniedException(name);
      case READ_SYMLINK_OR_JUNCTION_DOES_NOT_EXIST:
        throw new FileNotFoundException(name);
      case READ_SYMLINK_OR_JUNCTION_NOT_A_LINK:
        throw new NotASymlinkException(PathFragment.create(name));
      default:
        // This is READ_SYMLINK_OR_JUNCTION_ERROR (1). The JNI code puts a custom message in
        // 'error[0]'.
        throw new IOException(String.format("Cannot read link (name=%s): %s", name, error[0]));
    }
  }

  public static boolean deletePath(String path) throws IOException {
    String[] error = new String[] {null};
    int result = nativeDeletePath(WindowsPathOperations.asLongPath(path), error);
    switch (result) {
      case DELETE_PATH_SUCCESS:
        return true;
      case DELETE_PATH_DOES_NOT_EXIST:
        return false;
      case DELETE_PATH_DIRECTORY_NOT_EMPTY:
        throw new java.nio.file.DirectoryNotEmptyException(path);
      case DELETE_PATH_ACCESS_DENIED:
        throw new java.nio.file.AccessDeniedException(path);
      default:
        // This is DELETE_PATH_ERROR (1). The JNI code puts a custom message in 'error[0]'.
        throw new IOException(String.format("Cannot delete path '%s': %s", path, error[0]));
    }
  }
}
