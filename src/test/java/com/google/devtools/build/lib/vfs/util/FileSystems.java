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
package com.google.devtools.build.lib.vfs.util;

import com.google.common.base.Verify;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem;

/**
 * This static file system singleton manages access to a single default
 * {@link FileSystem} instance created within the methods of this class.
 */
@ThreadSafe
public final class FileSystems {

  private FileSystems() {}

  private static FileSystem defaultNativeFileSystem;
  private static FileSystem defaultJavaIoFileSystem;

  /**
   * Initializes the default native {@link FileSystem} instance as a platform native
   * (Unix or Windows) file system. If it's not initialized, then initialize it,
   * otherwise verify if the type of the instance is correct.
   */
  public static synchronized FileSystem getNativeFileSystem() {
    if (OS.getCurrent() == OS.WINDOWS) {
      if (defaultNativeFileSystem == null) {
        defaultNativeFileSystem = new WindowsFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
      } else {
        Verify.verify(defaultNativeFileSystem instanceof WindowsFileSystem);
      }
    } else {
      if (defaultNativeFileSystem == null) {
        try {
          defaultNativeFileSystem =
              (FileSystem)
                  Class.forName(TestConstants.TEST_REAL_UNIX_FILE_SYSTEM)
                      .getDeclaredConstructor(DigestHashFunction.class)
                      .newInstance(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
        } catch (Exception e) {
          throw new IllegalStateException(e);
        }
      } else {
        Verify.verify(defaultNativeFileSystem instanceof UnixFileSystem);
      }
    }
    return defaultNativeFileSystem;
  }

  /**
   * Initializes the default java {@link FileSystem} instance as a java.io.File
   * file system. If it's not initialized, then initialize it,
   * otherwise verify if the type of the instance is correct.
   */
  public static synchronized FileSystem getJavaIoFileSystem() {
    if (defaultJavaIoFileSystem == null) {
      defaultJavaIoFileSystem = new JavaIoFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
    } else {
      Verify.verify(defaultJavaIoFileSystem instanceof JavaIoFileSystem);
    }
    return defaultJavaIoFileSystem;
  }
}
