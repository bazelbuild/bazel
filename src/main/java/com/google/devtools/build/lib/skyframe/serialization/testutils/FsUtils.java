// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.testutils;

import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

/** Common FileSystem related items for serialization tests. */
public class FsUtils {

  public static final FileSystem TEST_FILESYSTEM = new InMemoryFileSystem();

  public static final FileSystemProvider TEST_FILESYSTEM_PROVIDER = () -> TEST_FILESYSTEM;

  public static final RootedPath TEST_ROOT =
      RootedPath.toRootedPath(
          TEST_FILESYSTEM.getPath(PathFragment.create("/anywhere/at/all")),
          PathFragment.create("all/at/anywhere"));

  private FsUtils() {}

  /** Returns path relative to {@link #TEST_ROOT}. */
  public static PathFragment rootPathRelative(String path) {
    return TEST_ROOT.getRelativePath().getRelative(path);
  }
}
