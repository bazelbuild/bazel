// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RecursivePkgKey}. */
@RunWith(JUnit4.class)
public class RecursivePkgKeyTest extends BuildViewTestCase {

  private SkyKey buildRecursivePkgKey(
      RepositoryName repository,
      PathFragment rootRelativePath,
      ImmutableSet<PathFragment> excludedPaths) {
    RootedPath rootedPath = RootedPath.toRootedPath(Root.fromPath(rootDirectory), rootRelativePath);
    return RecursivePkgValue.key(repository, rootedPath, new IgnoredSubdirectories(excludedPaths));
  }

  private void invalidHelper(
      PathFragment rootRelativePath, ImmutableSet<PathFragment> excludedPaths) {
    assertThrows(
        IllegalArgumentException.class,
        () -> buildRecursivePkgKey(RepositoryName.MAIN, rootRelativePath, excludedPaths));
  }

  @Test
  public void testValidRecursivePkgKeys() throws Exception {
    buildRecursivePkgKey(
        RepositoryName.MAIN,
        PathFragment.create(""),
        ImmutableSet.<PathFragment>of());
    buildRecursivePkgKey(
        RepositoryName.MAIN,
        PathFragment.create(""),
        ImmutableSet.of(PathFragment.create("a")));

    buildRecursivePkgKey(
        RepositoryName.MAIN,
        PathFragment.create("a"),
        ImmutableSet.<PathFragment>of());
    buildRecursivePkgKey(
        RepositoryName.MAIN,
        PathFragment.create("a"),
        ImmutableSet.of(PathFragment.create("a/b")));

    buildRecursivePkgKey(
        RepositoryName.MAIN,
        PathFragment.create("a/b"),
        ImmutableSet.<PathFragment>of());
    buildRecursivePkgKey(
        RepositoryName.MAIN,
        PathFragment.create("a/b"),
        ImmutableSet.of(PathFragment.create("a/b/c")));
  }

  @Test
  public void testInvalidRecursivePkgKeys() throws Exception {
    invalidHelper(PathFragment.create(""), ImmutableSet.of(PathFragment.create("")));
    invalidHelper(PathFragment.create("a"), ImmutableSet.of(PathFragment.create("a")));
    invalidHelper(PathFragment.create("a"), ImmutableSet.of(PathFragment.create("b")));
    invalidHelper(PathFragment.create("a/b"), ImmutableSet.of(PathFragment.create("a")));
  }
}
