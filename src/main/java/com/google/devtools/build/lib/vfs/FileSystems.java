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

/**
 * This static file system singleton manages access to a single default
 * {@link FileSystem} instance created within the methods of this class.
 */
@ThreadSafe
@Deprecated // Instantiate and inject FileSystem instances directly, or use
            // com.google.devtools.build.lib.vfs.util.FileSystems in tests.
public final class FileSystems {

  private FileSystems() {}

  private static FileSystem defaultFileSystem;

  /**
   * Initializes the default {@link FileSystem} instance as a platform native
   * (Unix) file system, creating one iff needed, and returns the instance.
   *
   * <p>This method is idempotent as long as the initialization is of the same
   * type (Native/JavaIo/Union).
   */
  public static synchronized FileSystem initDefaultAsNative() {
    if (!(defaultFileSystem instanceof UnixFileSystem)) {
      defaultFileSystem = new UnixFileSystem();
    }
    return defaultFileSystem;
  }

  /**
   * Initializes the default {@link FileSystem} instance as a java.io.File
   * file system, creating one iff needed, and returns the instance.
   *
   * <p>This method is idempotent as long as the initialization is of the same
   * type (Native/JavaIo/Union).
   */
  public static synchronized FileSystem initDefaultAsJavaIo() {
    if (!(defaultFileSystem instanceof JavaIoFileSystem)) {
      defaultFileSystem = new JavaIoFileSystem();
    }
    return defaultFileSystem;
  }
}
