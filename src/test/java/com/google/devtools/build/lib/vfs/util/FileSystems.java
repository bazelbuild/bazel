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

import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem;

/** Convenience factory methods. */
public final class FileSystems {

  private FileSystems() {}

  /** Constructs a platform native (Unix or Windows) file system. */
  public static FileSystem getNativeFileSystem(DigestHashFunction digestHashFunction) {
    if (OS.getCurrent() == OS.WINDOWS) {
      return new WindowsFileSystem(digestHashFunction, /* createSymbolicLinks= */ true);
    }
    try {
      return Class.forName(TestConstants.TEST_REAL_UNIX_FILE_SYSTEM)
          .asSubclass(FileSystem.class)
          .getDeclaredConstructor(DigestHashFunction.class, String.class)
          .newInstance(digestHashFunction, TestConstants.TEST_UNIX_HASH_ATTRIBUTE);
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  /** Constructs a platform native (Unix or Windows) file system with SHA256 digests. */
  public static FileSystem getNativeFileSystem() {
    return getNativeFileSystem(DigestHashFunction.SHA256);
  }

  /** Constructs a java.io.File file system. */
  public static FileSystem getJavaIoFileSystem() {
    return new JavaIoFileSystem(DigestHashFunction.SHA256);
  }
}
