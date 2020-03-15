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
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

/** Common FileSystem related items for serialization tests. */
public class FsUtils {

  public static final FileSystem TEST_FILESYSTEM = new InMemoryFileSystem();

  private static final Root TEST_ROOT =
      Root.fromPath(TEST_FILESYSTEM.getPath(PathFragment.create("/anywhere/at/all")));

  public static final RootedPath TEST_ROOTED_PATH =
      RootedPath.toRootedPath(TEST_ROOT, PathFragment.create("all/at/anywhere"));

  private FsUtils() {}

  /** Returns path relative to {@link #TEST_ROOTED_PATH}. */
  public static PathFragment rootPathRelative(String path) {
    return TEST_ROOTED_PATH.getRootRelativePath().getRelative(path);
  }

  public static void addDependencies(SerializationTester tester) {
    tester.addDependency(FileSystem.class, TEST_FILESYSTEM);
    tester.addDependency(
        Root.RootCodecDependencies.class,
        new Root.RootCodecDependencies(/*likelyPopularRoot=*/ TEST_ROOT));
  }
}
