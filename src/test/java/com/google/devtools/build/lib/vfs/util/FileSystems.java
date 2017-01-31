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
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnionFileSystem;
import com.google.devtools.build.lib.vfs.UnixFileSystem;
import com.google.devtools.build.lib.vfs.WindowsFileSystem;
import com.google.devtools.build.lib.vfs.ZipFileSystem;

import java.io.IOException;
import java.util.Map;

/**
 * This static file system singleton manages access to a single default
 * {@link FileSystem} instance created within the methods of this class.
 */
@ThreadSafe
public final class FileSystems {

  private FileSystems() {}

  private static FileSystem defaultNativeFileSystem;
  private static FileSystem defaultJavaIoFileSystem;
  private static FileSystem defaultUnionFileSystem;

  /**
   * Initializes the default native {@link FileSystem} instance as a platform native
   * (Unix or Windows) file system. If it's not initialized, then initialize it,
   * otherwise verify if the type of the instance is correct.
   */
  public static synchronized FileSystem getNativeFileSystem() {
    if (OS.getCurrent() == OS.WINDOWS) {
      if (defaultNativeFileSystem == null) {
        defaultNativeFileSystem = new WindowsFileSystem();
      } else {
        Verify.verify(defaultNativeFileSystem instanceof WindowsFileSystem);
      }
    } else {
      if (defaultNativeFileSystem == null) {
        try {
          defaultNativeFileSystem = (FileSystem)
              Class.forName(TestConstants.TEST_REAL_UNIX_FILE_SYSTEM).newInstance();
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
      defaultJavaIoFileSystem = new JavaIoFileSystem();
    } else {
      Verify.verify(defaultJavaIoFileSystem instanceof JavaIoFileSystem);
    }
    return defaultJavaIoFileSystem;
  }

  /**
   * Initializes the default union {@link FileSystem} instance as a
   * {@link UnionFileSystem}. If it's not initialized, then initialize it,
   * otherwise verify if the type of the instance is correct.
   *
   * @param prefixMapping the desired mapping of path prefixes to delegate file systems
   * @param rootFileSystem the default file system for paths that don't match any prefix map
   */
  public static synchronized FileSystem getUnionFileSystem(
      Map<PathFragment, FileSystem> prefixMapping, FileSystem rootFileSystem) {
    if (defaultUnionFileSystem == null) {
      defaultUnionFileSystem = new UnionFileSystem(prefixMapping, rootFileSystem);
    } else {
      Verify.verify(defaultUnionFileSystem instanceof UnionFileSystem);
    }
    return defaultUnionFileSystem;
  }

  /**
   * Returns a new instance of a simple {@link FileSystem} implementation that
   * presents the contents of a zip file as a read-only file system view.
   */
  public static FileSystem getZipFileSystem(Path zipFile) throws IOException {
    return new ZipFileSystem(zipFile);
  }
}
