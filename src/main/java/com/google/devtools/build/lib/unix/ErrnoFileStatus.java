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

import com.google.devtools.build.lib.jni.JniLoader;
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
  public static final int ENODATA;

  static {
    ErrnoConstants constants = ErrnoConstants.getErrnoConstants();
    ENOENT = constants.errnoENOENT;
    EACCES = constants.errnoEACCES;
    ELOOP = constants.errnoELOOP;
    ENOTDIR = constants.errnoENOTDIR;
    ENAMETOOLONG = constants.errnoENAMETOOLONG;
    ENODATA = constants.errnoENODATA;
  }

  /** Constructs a ErrnoFileSatus instance. (Called only from JNI code.) */
  private ErrnoFileStatus(
      int mode, long atime, long mtime, long ctime, long size, int dev, long ino) {
    super(mode, atime, mtime, ctime, size, dev, ino);
    this.errno = 0;
  }

  /**
   * Constructs a ErrnoFileSatus instance.  (Called only from JNI code.)
   */
  private ErrnoFileStatus(int errno) {
    super(0, 0, 0, 0, 0, 0, 0);
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
    private int errnoENOENT;
    private int errnoEACCES;
    private int errnoELOOP;
    private int errnoENOTDIR;
    private int errnoENAMETOOLONG;
    private int errnoENODATA;

    public static ErrnoConstants getErrnoConstants() {
      ErrnoConstants constants = new ErrnoConstants();
      if (OS.getCurrent() != OS.WINDOWS) {
        constants.initErrnoConstants();
      }
      return constants;
    }

    static {
      JniLoader.loadJni();
    }

    private native void initErrnoConstants();
  }
}
