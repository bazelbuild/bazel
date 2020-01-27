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

import com.google.devtools.build.lib.UnixJniLoader;
import com.google.devtools.build.lib.util.OS;

/**
 * A subsclass of FileStatus which contains an errno.
 * If there is an error, all other data fields are undefined.
 */
public class ErrnoFileStatus extends FileStatus {

  private final int errno;

  // These constants are passed in from JNI via ErrnoConstants.
  public static final int ENOENT;
  public static final int EACCES;
  public static final int ELOOP;
  public static final int ENOTDIR;
  public static final int ENAMETOOLONG;

  static {
    ErrnoConstants constants = ErrnoConstants.getErrnoConstants();
    ENOENT = constants.ENOENT;
    EACCES = constants.EACCES;
    ELOOP = constants.ELOOP;
    ENOTDIR = constants.ENOTDIR;
    ENAMETOOLONG = constants.ENAMETOOLONG;
  }

  /**
   * Constructs a ErrnoFileSatus instance.  (Called only from JNI code.)
   */
  private ErrnoFileStatus(int st_mode, int st_atime, int st_atimensec, int st_mtime,
                          int st_mtimensec, int st_ctime, int st_ctimensec, long st_size,
                          int st_dev, long st_ino) {
    super(st_mode, st_atime, st_atimensec, st_mtime, st_mtimensec, st_ctime, st_ctimensec, st_size,
          st_dev, st_ino);
    this.errno = 0;
  }

  /**
   * Constructs a ErrnoFileSatus instance.  (Called only from JNI code.)
   */
  private ErrnoFileStatus(int errno) {
    super(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    this.errno = errno;
  }

  public int getErrno() {
    return errno;
  }

  public boolean hasError() {
    // errno = 0 means the operation succeeded.
    return errno != 0;
  }

  // Used to transfer the constants from native to java code.
  private static class ErrnoConstants {

    // These are set in JNI.
    private int ENOENT;
    private int EACCES;
    private int ELOOP;
    private int ENOTDIR;
    private int ENAMETOOLONG;

    public static ErrnoConstants getErrnoConstants() {
      ErrnoConstants constants = new ErrnoConstants();
      if (OS.getCurrent() != OS.WINDOWS) {
        constants.initErrnoConstants();
      }
      return constants;
    }

    static {
      UnixJniLoader.loadJni();
    }

    private native void initErrnoConstants();
  }
}
