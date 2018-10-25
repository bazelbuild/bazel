// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExperimentalEventHandler} static methods. */
@RunWith(JUnit4.class)
public class ExperimentalEventHandlerTest {
  private final FileSystem fileSystem = new InMemoryFileSystem();

  @Test
  public void getRelativeLocationString_PathIsAlreadyRelative() {
    assertThat(
            ExperimentalEventHandler.getRelativeLocationString(
                Location.fromPathAndStartColumn(
                    PathFragment.create("relative/path"), 0, 0, new LineAndColumn(4, 2)),
                PathFragment.create("/this/is/the/workspace"),
                ImmutableList.of(
                    Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root")))))
        .isEqualTo("relative/path:4:2");
  }

  @Test
  public void getRelativeLocationString_PathIsAbsoluteAndWorkspaceIsNull() {
    assertThat(
            ExperimentalEventHandler.getRelativeLocationString(
                Location.fromPathAndStartColumn(
                    PathFragment.create("/absolute/path"), 0, 0, new LineAndColumn(4, 2)),
                null,
                ImmutableList.of(
                    Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root")))))
        .isEqualTo("/absolute/path:4:2");
  }

  @Test
  public void getRelativeLocationString_PathIsAbsoluteButNotUnderWorkspaceOrPackagePathRoots() {
    assertThat(
            ExperimentalEventHandler.getRelativeLocationString(
                Location.fromPathAndStartColumn(
                    PathFragment.create("/absolute/path"), 0, 0, new LineAndColumn(4, 2)),
                PathFragment.create("/this/is/the/workspace"),
                ImmutableList.of(
                    Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root")))))
        .isEqualTo("/absolute/path:4:2");
  }

  @Test
  public void getRelativeLocationString_PathIsAbsoluteAndUnderWorkspace() {
    assertThat(
            ExperimentalEventHandler.getRelativeLocationString(
                Location.fromPathAndStartColumn(
                    PathFragment.create("/this/is/the/workspace/blah.txt"),
                    0,
                    0,
                    new LineAndColumn(4, 2)),
                PathFragment.create("/this/is/the/workspace"),
                ImmutableList.of(
                    Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root")))))
        .isEqualTo("blah.txt:4:2");
  }

  @Test
  public void getRelativeLocationString_PathIsAbsoluteAndUnderPackagePathRoot() {
    assertThat(
        ExperimentalEventHandler.getRelativeLocationString(
            Location.fromPathAndStartColumn(
                PathFragment.create("/this/is/a/package/path/root3/blah.txt"),
                0,
                0,
                new LineAndColumn(4, 2)),
            PathFragment.create("/this/is/the/workspace"),
            ImmutableList.of(
                Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root1")),
                Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root2")),
                Root.fromPath(fileSystem.getPath("/this/is/a/package/path/root3")))))
        .isEqualTo("blah.txt:4:2");
  }
}
