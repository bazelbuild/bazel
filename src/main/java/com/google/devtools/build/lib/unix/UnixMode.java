// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.vfs.Dirent;

/** Constants and utilities for working with Unix file modes. */
public final class UnixMode {

  // Note: even though POSIX doesn't specify the concrete values of these constants, all Unix
  // implementations we care about (Linux, macOS and BSDs) agree on them.

  public static final int S_IFMT = 0170000; // mask: filetype bitfields
  public static final int S_IFSOCK = 0140000; // socket
  public static final int S_IFLNK = 0120000; // symbolic link
  public static final int S_IFREG = 0100000; // regular file
  public static final int S_IFBLK = 0060000; // block device
  public static final int S_IFDIR = 0040000; // directory
  public static final int S_IFCHR = 0020000; // character device
  public static final int S_IFIFO = 0010000; // fifo
  public static final int S_ISUID = 0004000; // set UID bit
  public static final int S_ISGID = 0002000; // set GID bit (see below)
  public static final int S_ISVTX = 0001000; // sticky bit (see below)
  public static final int S_IRWXA = 00777; // mask: all permissions
  public static final int S_IRWXU = 00700; // mask: file owner permissions
  public static final int S_IRUSR = 00400; // owner has read permission
  public static final int S_IWUSR = 00200; // owner has write permission
  public static final int S_IXUSR = 00100; // owner has execute permission
  public static final int S_IRWXG = 00070; // mask: group permissions
  public static final int S_IRGRP = 00040; // group has read permission
  public static final int S_IWGRP = 00020; // group has write permission
  public static final int S_IXGRP = 00010; // group has execute permission
  public static final int S_IRWXO = 00007; // mask: other permissions
  public static final int S_IROTH = 00004; // others have read permission
  public static final int S_IWOTH = 00002; // others have write permission
  public static final int S_IXOTH = 00001; // others have execute permission
  public static final int S_IEXEC = 00111; // owner, group, world execute

  /** Returns the {@link Dirent.Type} for the given mode. */
  public static Dirent.Type getDirentTypeFromMode(int mode) {
    if (isSpecialFile(mode)) {
      return Dirent.Type.UNKNOWN;
    } else if (isFile(mode)) {
      return Dirent.Type.FILE;
    } else if (isDirectory(mode)) {
      return Dirent.Type.DIRECTORY;
    } else if (isSymbolicLink(mode)) {
      return Dirent.Type.SYMLINK;
    } else {
      return Dirent.Type.UNKNOWN;
    }
  }

  /** Returns whether the mode represents a file, including a special file. */
  public static boolean isFile(int mode) {
    int type = mode & S_IFMT;
    return type == S_IFREG || isSpecialFile(mode);
  }

  /** Returns whether the mode represents a special file. */
  public static boolean isSpecialFile(int mode) {
    int type = mode & S_IFMT;
    return type == S_IFSOCK || type == S_IFBLK || type == S_IFCHR || type == S_IFIFO;
  }

  /** Returns whether the mode represents a directory. */
  public static boolean isDirectory(int mode) {
    int type = mode & S_IFMT;
    return type == S_IFDIR;
  }

  /** Returns whether the mode represents a symbolic link. */
  public static boolean isSymbolicLink(int mode) {
    int type = mode & S_IFMT;
    return type == S_IFLNK;
  }

  /** Returns the permissions of the mode. */
  public static int getPermissions(int mode) {
    return mode & S_IRWXA;
  }

  private UnixMode() {}
}
